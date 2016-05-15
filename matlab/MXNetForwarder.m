classdef MXNetForwarder < handle
    %MODEL MXNet model, supports load and forward
    
    properties
        
        % Model pointer
        pCtx
        % Symbol definition JSON
        SymbolJson;
        % Parameters weights file
        ParamsLen;
        ParamsPtr;
        
    end
    
    properties (Access = private)
        
    end
    
    methods
        function obj = MXNetForwarder(symblFile, paramFile)
            
            % CONSTRUCTOR
            obj.pCtx = libpointer('voidPtr', 0);
            assert(obj.pCtx.Value == 0);
            
            % Load model symbol JSON
            obj.SymbolJson = fileread(symblFile);
            
            % Load model params
            fid = fopen(paramFile, 'rb');
            Params = fread(fid, inf, '*ubit8');
            fclose(fid);
            obj.ParamsLen = length(Params);
            obj.ParamsPtr = libpointer('voidPtr', Params);
            
        end
        
        function obj = forward(obj, img)
            
            siz = size(img);
            assert(length(siz) >= 2);
            
            img = permute(img, [2 1 3:length(siz)]);
            
            X_data  = single(img(:)); % take cols and concat
            X_len   = uint32(numel(X_data));
            X_dim   = int32([ones(1, 4-length(siz)), siz(end:-1:1)]);
            X_shape = uint32([0, 4]);
            
            %% Run prediction
            fprintf('create predictor with input size ');
            fprintf('%d ', siz);
            fprintf('\n');
            
            callmxnet('MXPredCreate', ...
                obj.SymbolJson, ...
                obj.ParamsPtr, ...
                obj.ParamsLen, ...
                1, ...
                0, ...
                1, ...
                {'data'}, ...
                X_shape, ...
                X_dim, ...
                obj.pCtx);
            
            %% feed input
            callmxnet('MXPredSetInput', obj.pCtx, 'data', X_data, X_len);
            
            %% forward
            callmxnet('MXPredForward', obj.pCtx);
            
        end
        
        function pred = getOutput(obj, index)
            
            %% Get the output size and allocate pointer
            Y_dim   = libpointer('uint32Ptr', 0);
            Y_shape = libpointer('uint32PtrPtr', zeros(4,1));
            callmxnet('MXPredGetOutputShape', ...
                obj.pCtx, ...
                index, ...
                Y_shape, ...
                Y_dim);
            assert(Y_dim.Value <= 4);
            Y_size = Y_shape.Value(1:Y_dim.Value);
            Y_size = double(Y_size(end:-1:1))';
            
            %% Get the output daya
            Y_data = libpointer('singlePtr', single(zeros(Y_size)));
            
            callmxnet('MXPredGetOutput', ...
                obj.pCtx, ...
                index, ...
                Y_data, ...
                uint32(prod(Y_size)));
            
            % TODO convert from c order to matlab order...
            pred = reshape(Y_data.Value, Y_size);
            
            
        end
        
        function obj = free(obj)
            
            %% Free the model
            callmxnet('MXPredFree', obj.pCtx);
        end
        
    end
    
    methods (Access = private)
    end
    
end
