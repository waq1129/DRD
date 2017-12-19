#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <matrix.h>
#include <string.h>
#include "mex.h"

double compute_loss(double *X, double *y, double *lambda, double *beta, int n, int p);

int count_nnz_beta(double* beta, int p);

int sgn(double val);

void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[] )
{
	double *X = (double*) mxGetData(prhs[0]);
	double *y = (double*) mxGetData(prhs[1]);

	// lambda can be different for different coordinates
	double *lambda= (double*)mxGetData(prhs[2]);

	int p = mxGetN(prhs[0]);	// num of features
	int n = mxGetM(prhs[0]);	// num of instances

	int max_outer_its = (int)mxGetScalar(prhs[3]);

	// printf("n=%d\t p=%d\t it=%d\n", n, p, max_outer_its);

	plhs[0] = mxCreateDoubleMatrix(p, 1, mxREAL);
	double* beta = mxGetPr(plhs[0]);

	int i, j, k;
	int it;

	// compute the diag(X' * X)
	double *xx_diag = (double*) mxCalloc(p, sizeof(double));
	for (j = 0; j < p; ++j)
	{
		xx_diag[j] = 0;
		for (i = 0; i < n; ++i)
		{
			xx_diag[j] += X[j*n + i] * X[j*n + i];
		}
	}
	// y-X_{-i} beta_{-i}
	double *res = (double*) mxCalloc(n, sizeof(double));
	for (i = 0; i < n; ++i)
	{
		res[i] = y[i];
		// ignore the first coordinate
		for (j=1; j < p; ++j)
		{
			res[i] -= X[j * n + i] * beta[j];
		}
	}

	for (it = 0; it < max_outer_its; ++it)
	{
		for (j = 0; j < p; ++j)
		{
			beta[j] = 0;
			for (i = 0; i < n; ++i)
			{
				beta[j] += res[i] * X[j * n + i];
			}
			beta[j] /= xx_diag[j];
			if (fabs(beta[j]) <= lambda[j])
			{
				//printf("%f\t%f\n", beta[j], lambda[j]);
				beta[j] = 0;
			} else
			{
				beta[j] = sgn(beta[j]) * (fabs(beta[j]) - lambda[j]);
			}

			// update the residuals
			int next_coor = j+1;
			if (next_coor == p)
			{
				next_coor = 0;
			}
			for (i = 0; i < n; ++i)
			{
				res[i] += beta[next_coor] * X[next_coor * n + i];
				res[i] -= beta[j] * X[j * n + i];
			}
		}
		if (it % 10 == 0)
		{
			//printf("%d\n", count_nnz_beta(beta, p));
			//printf("%d\t%f\n", it, compute_loss(X, y, lambda, beta, n, p));
		}
	}
	//printf("%d\t%f\n", it, compute_loss(X, y, lambda, beta, n, p));
	mxFree(xx_diag);
	mxFree(res);
}

int count_nnz_beta(double* beta, int p)
{
	int i;
	int cnt = 0;
	for (i = 0; i < p; ++i)
	{
		cnt += (beta[i] == 0);
	}
	return cnt;

}
double compute_loss(double *X, double *y, double *lambda, double *beta, int n, int p)
{
	double loss = 0;
	int i, j;
	for (i = 0; i < n; ++i)
	{
		double pred = 0;
		for (j = 0; j < p; ++j)
		{
			pred += X[j * n + i] * beta[j];
		}
		//printf("%f\n", 0.5*(y[i] - pred)*(y[i] - pred));
		loss += 0.5*(y[i] - pred)*(y[i] - pred);
	}
	for (j = 0; j < p; ++j)
	{
		loss += lambda[j] * fabs(beta[j]);
	}
	return loss;
}

int sgn(double val)
{
	if (val > 0)
		return 1;
	else if(val < 0)
		return -1;
	else
		return 0;
}
