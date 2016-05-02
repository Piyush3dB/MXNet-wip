function callmxnet(func, varargin)
%CALLMXNET call mxnet functions

assert(ischar(func));
disp(['++Calling ' func ]);
ret = calllib('libmxnet', func, varargin{:});
%ret

assert(ret == 0);

end
