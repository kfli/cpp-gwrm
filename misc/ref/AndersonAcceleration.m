%------------------------------------------------------------------------
% Function yout = AndersonAcceleration(y,MP,AANUMERICS).
%
% Fixed-point iteration with or without Anderson acceleration.
% 'g' is the fixed point iteration map and 'y' is the 
% solution, so that y = g(y).
%
% Input arguments:
%    y = Initial value solution. Note: In this function this variable 
%        must be a column vector.
%    MP = struct with parameters required in 'g'.
%    AANUMERICS = Struct with various variables for Anderson Acceleration scheme.
%       1. 'mMax' = maximum number of stored residuals (non-negative integer).
%       NOTE: 'mMax' = 0 => no acceleration. default=1000
%       2. 'itmax' = maximum allowable number of iterations. default=1000
%       3. 'atol' = absolute error tolerance. default=1.0e-6
%       4. 'rtol' = relative error tolerance. default=1.0e-3
%       5. 'droptol' = tolerance for dropping stored residual vectors to 
%       improve conditioning: If 'droptol' > 0, drop residuals if the
%       condition number exceeds droptol; if droptol <= 0,
%       do not drop residuals.
%       6. 'beta' = damping factor: If 'beta' > 0 (and beta ~= 1), then 
%       the step is damped by beta; otherwise, the step is not damped.
%       NOTE: 'beta' can be a function handle; form beta(iter), where iter 
%       is the iteration number and 0 < beta(iter) <= 1.
%       7. 'AAstart' = acceleration delay factor: If 'AAstart' > 0, start 
%       acceleration when iter = AAstart.
%       8. 'plotconv' is a boolean, if true then it will activate "plot 
%       while solving". 
%       9. 'update' is a number > 1, that sets the update frequency of the 
%       plot while solving window.
%
% Output:
% yout = Solution vector.
% LOG = Struct with data from Anderson Acceleration:
%       err - Boolean. If false: converged, If true: not converged.
%       iterations - Number of iterations
%       residual - Residual of problem when converged
%       tolerance - Tolerance criterion
%
% The notation used is that of H.F. Walker: Anderson Acceleration:
% Algorithms and implementation
%  if ~isfield(INPUT,'mMax'),INPUT.mMax=1000;end
%  if ~isfield(INPUT,'itmax'),INPUT.itmax=100;end
%  if ~isfield(INPUT,'rtol'),INPUT.rtol=1e-3;end
%  if ~isfield(INPUT,'atol'),INPUT.atol=1e-6;end
%  if ~isfield(INPUT,'droptol'),INPUT.droptol=1e8;end
%  if ~isfield(INPUT,'beta'),INPUT.beta=1;end
%  if ~isfield(INPUT,'AAstart'),INPUT.AAstart=0;end
%------------------------------------------------------------------------
%function  [yout,LOG] = AndersonAcceleration(y,MP,AANUMERICS)
function  [yout,LOG] = AndersonAcceleration(tm,N,M,y,MP,AANUMERICS,c)

  % Checking input arguments
  y = y(:); % Must be column vector
  LOG.err = true; % For convergence, err must turn to false at the end

  % Extracting AA parameters from AANUMERICS
  mMax    = AANUMERICS.mMax;
  itmax   = AANUMERICS.itmax;
  atol    = AANUMERICS.atol;
  rtol    = AANUMERICS.rtol;
  droptol = AANUMERICS.droptol;
  beta    = AANUMERICS.beta;
  AAstart = AANUMERICS.AAstart;

  % Checking AA parameters
  if mMax <0,error('AA.m: mMax must be a positive integer');end
  if itmax <0,error('AA.m: itmax must be a positive integer');end
  if atol <0 || rtol <0,error('AA.m: Tolerances atol and rtol must be positive');end
  if droptol <0,error('AA.m: droptol must be positive');end
  if beta <0 || beta>1,error('AA.m: beta cannot be smaller than 0 or larger than 1');end
  if AAstart <0,error('AA.m: AAstart must be positive');end

  % Initialize storage arrays and number of stored residuals.
  %res_hist = [];
  DG = [];
  mAA = 0;

  % Printing parameters
  if mMax == 0
    %fprintf('\n No Anderson acceleration.\n');
  elseif mMax > 0
    %fprintf('\n mMax = %d \n',mMax);
  else
    %error('mMax must positive.');
  end

  %fprintf(' Iterations: %g\n', itmax);
  if beta~=1 && isnumeric(beta)
    %fprintf(' Damping: %g \n',beta);
  elseif beta == 1
    %fprintf(' Damping: No damping\n')
  else
    %fprintf(' Damping: Determined by function\n')
  end
  % --------------------------------------------------------
  %             ANDERSON ACCELERATION ITERATIONS
  % --------------------------------------------------------
  %fhist = []; %% REMOVE
  for iter = 0:itmax
    MP.iter = iter;
    a = gpuArray(y);
    gval = gather(matlab_test(tm,N,M,a,c));
    fval = gval - y;
    res_norm = norm(fval);
	
    % Set the residual tolerance on the initial iteration.
    if iter == 0
      tol = max(atol,rtol*res_norm);
      %fprintf(' Residual tolerance; %.2e\n', tol);
      %fprintf('\n Iter  Res.norm  Conv%% \n');
    end

    % Printing convergence data
    %convergence = min ( 100 * ( log10(  max(res_norm/tol,1) ) )^(-1) , 100);      
    %fprintf(' %d  %.2e  %2.1f%% \n', iter, res_norm, convergence);
    
    % Residual norm history
    %res_hist = [res_hist;[iter,res_norm]];
    
    % Convergence test, if converged the loop stops.
    if res_norm <= tol
      % Update plot while solving
      %figure(fig);
      %plotconvdata(gval,MP,res_norm);

      % Writing data to LOG struct
      LOG.err = false;
      LOG.res_norm = res_norm;
      LOG.iterations = iter;
      LOG.tolerance = tol;
      LOG.AANUMERICS = AANUMERICS;
      
      % Printing data to screen
      %fprintf('\n Problem converged');
      %fprintf('\n Iterations: %d',iter);
      %fprintf('\n Residual norm: %.2e', res_norm);
      %fprintf('\n Tolerance: %.2e\n',tol);
      break;   % Breaks for-loop
    end
    
    % If resnorm is larger than 1e8 at iter > 5, problem stops
    if res_norm >1e8 && iter > 5
      %fprintf('\nProblem diverging, stopping solver\n');
      break; % Breaks for-loop, diverged
    end
    
    % Fixed point iteration without acceleration, if mMax == 0.
    if mMax == 0 || iter < AAstart
      % We update E <- g(E) to obtain the next approximate solution.
      y = gval;
    else
      % With Anderson acceleration.
      % Update the df vector and the DG array.
      if iter > AAstart
        df = fval-f_old;
        if mAA < mMax
          DG = [DG gval-g_old];
        else
          DG = [DG(:,2:mAA) gval-g_old];
        end
        mAA = mAA + 1;
      end   % iter
        
      % We define the old g and f values for the next iteration
      f_old = fval;
      g_old = gval;
        
      if mAA == 0
        % Initialization
        % If mAA == 0, update y <- g(y) to obtain themah next approximate
        % solution. No least-squares problem is solved for iter = 0
        y = gval;
      else
        % If mAA > 0 we solve the least-squares problem and update the
        % solution.
        if mAA == 1
          % We form the initial QR decomposition.
          R(1,1) = norm(df);
          Q = R(1,1)\df;
        else
          % If mAA > 1, update the QR decomposition.   
          if mAA > mMax
            % If the column dimension of Q is mMax, delete the 
            % first column and update the decomposition.
            [Q,R] = qrdelete(Q,R,1);
            mAA = mAA - 1;
            % The following treats the qrdelete quirk described below.
            if size(R,1) ~= size(R,2)
              Q = Q(:,1:mAA-1); R = R(1:mAA-1,:);
            end
            % Explanation: If Q is not square, then qrdelete(Q,R,1)
            % reduces the column dimension of Q by 1 and the column
            % and row dimensions of R by 1. But if Q *is* square, 
            % then the column dimension of Q is not reduced and 
            % only the column dimension of R is reduced by one. 
            % This is to allow for MATLAB's default "thick" QR 
            % decomposition, which always produces a square Q.
          end  % mAA > mMax
          % Now update the QR decomposition to incorporate the new
          % column.
          for j = 1:mAA - 1
            R(j,mAA) = Q(:,j)'*df;
            df = df - R(j,mAA)*Q(:,j);
          end
          R(mAA,mAA) = norm(df);
          Q = [Q,R(mAA,mAA)\df];
        end  % mAA == 1
            
        if droptol > 0
          % Drop residuals to improve conditioning if necessary.
       
          condDF = 1/rcond(R);
          while condDF > droptol && mAA > 1
            %fprintf(' cond(D) = %e, reducing mAA to %d \n', condDF, mAA-1);
            [Q,R] = qrdelete(Q,R,1);
            DG = DG(:,2:mAA);
            mAA = mAA - 1;
            % The following treats the qrdelete quirk described 
            % above.
            if size(R,1) ~= size(R,2)
              Q = Q(:,1:mAA); R = R(1:mAA,:);
            end
            condDF = 1/rcond(R);    
          end  % While
        end  % droptol
            
        % Solve the least-squares problem.
        gamma = R\(Q'*fval);
        % Update the approximate solution.
        y = gval - DG*gamma;
        % Apply damping if beta is a function handle or if beta > 0
        % (and beta ~= 1).
          
        % Damping for non-zero beta
        if isa(beta,'function_handle')
          y = y - (1-beta(iter))*(fval - Q*R*gamma);
        else
          if beta > 0 && beta ~= 1
            y = y - (1-beta)*(fval - Q*R*gamma);
          end
        end % isa(beta ...
      end % mAA = 0
    end  % mMax == 0
  end  % For loop end

  %% Terminating the fixed point iteration loop.
  % Printing no of iterations, res.norm and tol.
  %if res_norm > tol && iter == itmax
  %  fprintf('\n Terminate after itmax: %d iterations.', itmax);
  %  fprintf('\n Residual norm: %.2e', res_norm);
  %  fprintf('\n Tolerance: %.2e\n',tol);
  %end
  iter
  res_norm
  %% Returning solution
  yout = y;
end % Anderson Acceleration    
