function particle = slice_sweep(particle, slice_fn, sigma, step_out, DD)
%SLICE_SWEEP one set of axis-aligned slice-sampling updates of particle.pos
%
%     particle = slice_sweep(particle, slice_fn[, sigma[, step_out]])
%
% The particle position is updated with a standard univariate slice-sampler.
% Stepping out is linear (if step_out is true), but shrinkage is exponential. A
% sensible strategy is to set sigma conservatively large and turn step_out off.
% If it's hard to set a good sigma though, you should leave step_out=true.
%
% Inputs:
%     particle   sct   Structure contains:
%                              .pos - initial position on slice as Dx1 vector
%                                     (or any array)
%                           .Lpstar - log probability of .pos (up to a constant)
%                         .on_slice - needn't be set initially but is set
%                                     during slice sampling. Particle must enter
%                                     and leave this routine "on the slice".
%     slice_fn   @fn   particle = slice_fn(particle, Lpstar_min)
%                      If particle.on_slice then particle.Lpstar should be
%                      correct, otherwise its value is arbitrary.
%        sigma (D|1)x1 step size parameter(s) (default=1)
%     step_out   1x1   if non-zero, do stepping out procedure (default), else
%                      only step in (saves on fn evals, but takes smaller steps)
%
% Outputs:
%     particle   sct   particle.pos and .Lpstar are updated.

% Originally based on pseudo-code in David MacKay's text book p375
% Iain Murray, May 2004, January 2007, June 2008, January 2009

if nargin < 3; sigma = 1; end
if nargin < 4; step_out = 1; end
if nargin < 5;
    DD = numel(particle.pos);
end
if length(sigma) == 1
    sigma = repmat(sigma, length(DD), 1);
end
% A random order is more robust generally and important inside
% algorithms like nested sampling and AIS

randid = randperm(length(DD));
for idd = randid
    dd = DD(idd);
    particle.pid = particle.id;
    particle.id = dd; % which index of hyp for slice sampling
    epsgap = log(rand);
    Lpstar_min = particle.Lpstar + epsgap;
    
    % Create a horizontal interval (x_l, x_r) enclosing x_cur
    x_cur = particle.pos(dd);
    rr = rand;
    x_l = x_cur - rr*sigma(idd);
    x_r = x_cur + (1-rr)*sigma(idd);
    %%%%%%%%%%%%%%%%%%%%%%%
    %     x_cur
    %     %         if abs(x_cur)>6
    %     %         keyboard
    %     %         end
    %     x_l
    %     x_r
    %pause
    
    %     if dd==2
    % %         keyboard;
    %         logdrnge = .1; % range to explore in each direction
    %     logd0 = x_cur; % center of range;
    %     npts = 50; % number of grid points
    %     logdgrid = logd0+logdrnge*linspace(-1,1,npts); % grid of logd values
    %     subplot(326)
    %     f_theta=[];
    %     pp = particle;
    %     for jj= 1:length(logdgrid);
    %         pp.pos(dd) = logdgrid(jj);
    %         pp = slice_fn(pp, Lpstar_min);
    %         f_theta = [f_theta; pp.Lpstar];
    %     end
    %     % plot(logdgrid, exp(f_theta-max(f_theta)),'-')
    %     plot(logdgrid, f_theta,'-')
    %     title(['dd=' num2str(dd)])
    %     drawnow, keyboard%(0.1)
    %     end
    %%%%%%%%%%%%%%%%%%%%%%%
    
    if step_out
        particle.pos(dd) = x_l;
        while 1
            %             display('low step_out!')
            particle = slice_fn(particle, Lpstar_min);
            if ~particle.on_slice
                break
            end
            particle.pos(dd) = particle.pos(dd) - sigma(idd);
        end
        %pause
        x_l = particle.pos(dd);
        particle.pos(dd) = x_r;
        while particle.pos(dd)>x_l
            %             display('up step_out!')
            particle = slice_fn(particle, Lpstar_min);
            if ~particle.on_slice
                break
            end
            particle.pos(dd) = particle.pos(dd) + sigma(idd);
        end
        x_r = particle.pos(dd);
        %pause
    end
    
    % Make proposals and shrink interval until acceptable point found
    % One should only get stuck in this loop forever on badly behaved problems,
    % which should probably be reformulated.
    chk = 0;
    while 1
        %         display('select in between!')
        rndnum = rand*(x_r - x_l);
        %         fprintf('rndnum:  %12.12f\n',rndnum);
        %         display(['pos0: ' num2str((pos0))]);
        newpos = rndnum + x_l;
        %         if abs(newpos-particle.pos(dd))<1e-6
        %             fprintf('stopped because newpos-particle is small: line 87\n');
        %             keyboard;
        %         end
        %             particle.pos(dd) = x_cur;
        %             particle.Lpstar = Lpstar_min;
        %             particle = slice_fn(particle, Lpstar_min);
        %
        %             break;
        %         end
        particle.pos(dd) = newpos;
        %         display(['new proposed pos: ' num2str(particle.pos)]);
        
        particle = slice_fn(particle, Lpstar_min);
        if particle.on_slice
            break % Only way to leave the while loop.
        else
            % Shrink in
            if particle.pos(dd) > x_cur
                x_r = particle.pos(dd);
            elseif particle.pos(dd) < x_cur
                x_l = particle.pos(dd);
            else
                error('BUG DETECTED: Shrunk to current position and still not acceptable.');
            end
        end
    end
    %pause
end

