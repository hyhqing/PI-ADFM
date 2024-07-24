function f = ShapeAwareFiltering(I, model,  params)
%
% Input:
%    I                         ����ͼ��
%    model                 ���ɭ�ֱ�Ե̽���㷨ģ�ͣ�
%    params               �������п��Ʋ�����
%Output:
%    f                         �˲���ͼ��

if(isfield(params, 'niter'))
    niter = params.niter;
else
    niter = 1;
end
if(isfield(params, 'BoxFilterSize'))
    k = params.BoxFilterSize;
else
    k = 1;
end
if(isfield(params, 'isShowGuiandce'))
    isShowGuiandce = params.isShowGuiandce;
else
    isShowGuiandce = false;
end
if(isfield(params, 'isShowFilt'))
    isShowFilt = params.isShowFilt;
else
    isShowFilt = false;
end

    f = gpuArray(I);  
    for  iter = 1:niter
        B = BoxFilterGPU(f, k);
        E = gpuArray(edgesDetect(gather(B), model));
        imwrite(gather(E), './overall/E.jpg')
        [G, ~] = GetGuidanceImgGPU(E, f, params);
        f = GuidanceFilter(f, G, params);
        if(isShowGuiandce)
            figure, imshow(G);
            imwrite(gather(G), ['./SingleTestRlt/G_iter = ', num2str(iter), '.jpg']);
        end
        if(isShowFilt)
            figure, imshow(f);
            imwrite(gather(f), ['./SingleTestRlt/f_iter = ', num2str(iter), '.jpg']);
        end
    end
    f = gather(f);
end