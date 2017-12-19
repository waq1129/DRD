function [act, alpha_out] = determine_alpha(q,s,gam_a,Used,M_full,zerofactor,priortype)
REEST = 0;
DEL = 1;
ADD = 2;
NONE = 3;

q2 = q.^2;
s2 = s.^2;

switch priortype
    case 'none'
        act = zeros(M_full,1); % re-estimate=0; delete=1; add=2; none=3;
        in = false(M_full,1);
        in(Used) = true;
        alpha_out = inf*ones(M_full,1);
        
        theta = q2-s;
        thetag0 = theta>zerofactor;
        alpha_new = s2./theta;
        % theta>0, alpha is in, act=REEST
        act(thetag0&in) = REEST;
        alpha_out(thetag0&in) = alpha_new(thetag0&in);
        % theta<=0, alpha is in, act=DEL
        act((~thetag0)&in) = DEL;
        % theta>0, alpha is out, act=ADD
        act(thetag0&(~in)) = ADD;
        alpha_out(thetag0&(~in)) = alpha_new(thetag0&(~in));
        % theta<=0, alpha is out, act=NONE
        act((~thetag0)&(~in)) = NONE;
        
    case 'gamma'
        lambda = gam_a+0.5;
        act = zeros(M_full,1); % re-estimate=0; delete=1; add=2; none=3;
        in = false(M_full,1);
        in(Used) = true;
        alpha_out = inf*ones(M_full,1);
        
        % L = (2*lambda-1)*alpha.^2+((4*lambda-1)*s-q2).*alpha+2*lambda*s2;
        % L = alpha.^2+b.*alpha+c;
        b = ((4*lambda-1)*s-q2)./(2*lambda-1);
        c = 2*lambda/(2*lambda-1).*s2;
        r = c-b.^2/4;
        rge0 = r>=0;
        rl0 = r<0;
        x1 = -sqrt(-r)-b/2;
        x2 = sqrt(-r)-b/2;
        x1le0 = x1<zerofactor;
        x2le0 = x2<zerofactor;

        if (2*lambda-1)>zerofactor
            % alpha=inf, alpha is in, act=DEL
            act(rge0&in) = DEL;
            % alpha=inf, alpha is out, act=NONE
            act(rge0&(~in)) = NONE;
            % alpha=inf, alpha is in, act=DEL
            act(rl0&x1le0&in) = DEL;
            % alpha=x1, alpha is in, act=REEST
            act(rl0&(~x1le0)&in) = REEST;
            alpha_out(rl0&(~x1le0)&in) = x1(rl0&(~x1le0)&in);
            % alpha=inf, alpha is out, act=NONE
            act(rl0&x1le0&(~in)) = NONE;
            % alpha=x1, alpha is out, act=ADD
            act(rl0&(~x1le0)&(~in)) = ADD;
            alpha_out(rl0&(~x1le0)&(~in)) = x1(rl0&(~x1le0)&(~in));
        elseif (2*lambda-1)<-zerofactor
            % alpha=inf, alpha is in, act=DEL
            act(rge0&in) = DEL;
            % alpha=inf, alpha is out, act=NONE
            act(rge0&(~in)) = NONE;
            % alpha=inf, alpha is in, act=DEL
            act(rl0&x2le0&in) = DEL;
            % alpha=x2, alpha is in, act=REEST
            act(rl0&(~x2le0)&in) = REEST;
            alpha_out(rl0&(~x2le0)&in) = x2(rl0&(~x2le0)&in);
            % alpha=inf, alpha is out, act=NONE
            act(rl0&x2le0&(~in)) = NONE;
            % alpha=x2, alpha is out, act=ADD
            act(rl0&(~x2le0)&(~in)) = ADD;
            alpha_out(rl0&(~x2le0)&(~in)) = x2(rl0&(~x2le0)&(~in));
        else
            theta = q2-(4*lambda-1)*s;
            thetag0 = theta>zerofactor;
            alpha_new = 2*lambda*s2./theta;
            % theta>0, alpha is in, act=REEST
            act(thetag0&in) = REEST;
            alpha_out(thetag0&in) = alpha_new(thetag0&in);
            % theta<=0, alpha is in, act=DEL
            act((~thetag0)&in) = DEL;
            % theta>0, alpha is out, act=ADD
            act(thetag0&(~in)) = ADD;
            alpha_out(thetag0&(~in)) = alpha_new(thetag0&(~in));
            % theta<=0, alpha is out, act=NONE
            act((~thetag0)&(~in)) = NONE;
        end
    case 'exp'
        act = zeros(M_full,1); % re-estimate=0; delete=1; add=2; none=3;
        in = false(M_full,1);
        in(Used) = true;
        alpha_out = inf*ones(M_full,1);
        m = gam_a;
        
        % L = (s-q2+2*m)*alpha.^2+(s2+4*m*s).*alpha+2*m*s2;
        % L = alpha.^2+b.*alpha+c;
        aa = s-q2+2*m;
        b = (s2+4*m*s)./aa;
        c = (2*m*s2)./aa;
        r = c-b.^2/4;
        rge0 = r>=0;
        rl0 = r<0;
        x1 = -sqrt(-r)-b/2;
        x2 = sqrt(-r)-b/2;
        x1le0 = x1<zerofactor | x1>1/zerofactor;
        x2le0 = x2<zerofactor | x2>1/zerofactor;
        
        aag0 = aa>zerofactor;
        aal0 = aa<-zerofactor;
        aae0 = (~aag0)&(~aal0);
        
        %         if aa>1e-2
        % alpha=inf, alpha is in, act=DEL
        act(aag0&rge0&in) = DEL;
        % alpha=inf, alpha is out, act=NONE
        act(aag0&rge0&(~in)) = NONE;
        % alpha=inf, alpha is in, act=DEL
        act(aag0&rl0&x1le0&in) = DEL;
        % alpha=x1, alpha is in, act=REEST
        act(aag0&rl0&(~x1le0)&in) = REEST;
        alpha_out(aag0&rl0&(~x1le0)&in) = x1(aag0&rl0&(~x1le0)&in);
        % alpha=inf, alpha is out, act=NONE
        act(aag0&rl0&x1le0&(~in)) = NONE;
        % alpha=x1, alpha is out, act=ADD
        act(aag0&rl0&(~x1le0)&(~in)) = ADD;
        alpha_out(aag0&rl0&(~x1le0)&(~in)) = x1(aag0&rl0&(~x1le0)&(~in));
        
        %         elseif aa<-1e-2
        % alpha=inf, alpha is in, act=DEL
        act(aal0&rge0&in) = DEL;
        % alpha=inf, alpha is out, act=NONE
        act(aal0&rge0&(~in)) = NONE;
        % alpha=inf, alpha is in, act=DEL
        act(aal0&rl0&x2le0&in) = DEL;
        % alpha=x2, alpha is in, act=REEST
        act(aal0&rl0&(~x2le0)&in) = REEST;
        alpha_out(aal0&rl0&(~x2le0)&in) = x2(aal0&rl0&(~x2le0)&in);
        % alpha=inf, alpha is out, act=NONE
        act(aal0&rl0&x2le0&(~in)) = NONE;
        % alpha=x2, alpha is out, act=ADD
        act(aal0&rl0&(~x2le0)&(~in)) = ADD;
        alpha_out(aal0&rl0&(~x2le0)&(~in)) = x2(aal0&rl0&(~x2le0)&(~in));
        
        %         else aa=0
        theta = s-2*q2;
        thetag0 = abs(theta)>zerofactor;
        alpha_new = (s.*(q2-s))./theta;
        pos = alpha_new>0;
        % theta>0, alpha is +, alpha is in, act=REEST
        act(aae0&thetag0&pos&in) = REEST;
        alpha_out(aae0&thetag0&pos&in) = alpha_new(aae0&thetag0&pos&in);
        % theta=0, alpha is +, alpha is in, act=DEL
        act(aae0&(~thetag0)&pos&in) = DEL;
        % theta>0, alpha is +, alpha is out, act=ADD
        act(aae0&thetag0&pos&(~in)) = ADD;
        alpha_out(aae0&thetag0&pos&(~in)) = alpha_new(aae0&thetag0&pos&(~in));
        % theta=0, alpha is +, alpha is out, act=NONE
        act(aae0&(~thetag0)&pos&(~in)) = NONE;
        % theta >=0, alpha is -, alpha is in, act=DEL
        act(aae0&(~pos)&in) = DEL;
        % theta >=0, alpha is -, alpha is out, act=NONE
        act(aae0&(~pos)&(~in)) = NONE;
end
end


