function callmxnet(func, varargin)
%CALLMXNET call mxnet functions

assert(ischar(func));
disp(['++Calling ' func ]);
ret = calllib('libmxnet', func, varargin{:});
assert(ret == 0);

end
