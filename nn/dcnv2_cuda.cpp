#include <dependency/depend.h>

// CUDA forward declarations
//#include <nn/dcnv2_cuda_kernel.cu>
// template <typename DType>
void modulated_deformable_im2col(
                  const at::Tensor data_im, const at::Tensor data_offset, const at::Tensor data_mask,
                                    const at::IntList& im_shape, const at::IntList& col_shape, const at::IntList& kernel_shape,
                                    const at::IntList& pad, const at::IntList& stride, const at::IntList& dilation,
                                    const int64_t deformable_group, at::Tensor data_col);
// template <typename DType>
void modulated_deformable_col2im(
                                    const at::Tensor data_col, const at::Tensor data_offset, const at::Tensor data_mask,
                                    const at::IntList& im_shape, const at::IntList& col_shape, const at::IntList& kernel_shape,
                                    const at::IntList& pad, const at::IntList& stride,
                                    const at::IntList& dilation, const int64_t deformable_group,
                                    at::Tensor grad_im);
// template <typename DType>
void modulated_deformable_col2im_coord(
                                        const at::Tensor data_col, const at::Tensor data_im, const at::Tensor data_offset, const at::Tensor data_mask,
                                        const at::IntList& im_shape, const at::IntList& col_shape, const at::IntList& kernel_shape,
                                        const at::IntList& pad, const at::IntList& stride,
                                        const at::IntList& dilation, const int64_t deformable_group,
                                        at::Tensor grad_offset, at::Tensor grad_mask);
// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIMS(x) AT_ASSERTM(x, #x " dimension wrong")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor>dcnv2_forward(
        at::Tensor im,
        at::Tensor weights,
        at::Tensor Offset,
        at::Tensor Mask,
        int64_t num_deformable_group,
        int64_t kernel,
        int64_t pad,
        int64_t stride,
        int64_t dilation,
        bool use_bias,
        at::Tensor bias
        )
        /*
        :param im: N,C,H,W
        :param weights: C`,C*kernel*kernel
        :param Offset: N,num_deformable_group*kernel*kernel*2,H,W
        :param Mask: N,num_deformable_group*kernel*kernel,H,W
        :param num_deformable_group: int , default 1
        :param kernel: int64_t
        :param pad: int64_t
        :param stride: int64_t
        :param dilation: int64_t
        :param use_bias : use bias or not
        :param bias: C` or None
        :return:
            std::vector<at::Tensor> [out,col]
                out : N, C', H', W' # H'= (H+2p-k)/s + 1 , W'= (W+2p-k)/s + 1
                col : C*kernel*kernel,NHW
        */
{
    // maybe not need check contiguous
    CHECK_INPUT(weights);
    if (use_bias) {
        CHECK_INPUT(bias);
    }

    // CHECK Tensor Shape
    CHECK_DIMS(im.dim()==4);
    CHECK_DIMS(weights.dim()==2);
    if (use_bias) {
        CHECK_DIMS(bias.dim()==1);
    }
    CHECK_DIMS(Offset.dim()==4);
    CHECK_DIMS(Mask.dim()==4);

    // Now we already support any size kernel/stride/pad/dilation conv
    //CHECK_DIMS (kernel == 3) ;
    //CHECK_DIMS (pad == 1 ) ;
    //CHECK_DIMS (stride == 1) ;
    //CHECK_DIMS (dilation == 1) ;

    at::IntList im_shape = im.sizes();
    int64_t im_batch = im_shape[0];
    int64_t im_channels = im_shape[1];
    int64_t im_H = im_shape[2];
    int64_t im_W = im_shape[3];

    at::IntList offset_shape = Offset.sizes();
    int64_t offset_batch = offset_shape[0];
    int64_t offset_channels = offset_shape[1];
    int64_t offset_H = offset_shape[2];
    int64_t offset_W = offset_shape[3];

    at::IntList mask_shape = Mask.sizes();
    int64_t mask_batch = mask_shape[0];
    int64_t mask_channels = mask_shape[1];
    int64_t mask_H = mask_shape[2];
    int64_t mask_W = mask_shape[3];

    at::IntList weight_shape = weights.sizes();
    int64_t weight_outchannels = weight_shape[0];
    int64_t weight_inchannels = weight_shape[1];

    at::IntList bias_shape ;
    int64_t bias_outchannels;
    if (use_bias){
        bias_shape = bias.sizes();
      bias_outchannels = bias_shape[0];
    }

    const int64_t N = im_batch ;
    const int64_t channels = im_channels ;
    const int64_t filters = weight_outchannels ;
    const int64_t H_IN = im_H ;
    const int64_t W_IN = im_W ;
    const int64_t H_OUT = (H_IN+2*pad-((kernel-1)*dilation+1))/stride + 1 ;
    const int64_t W_OUT = (W_IN+2*pad-((kernel-1)*dilation+1))/stride + 1 ;
    CHECK_DIMS(im_batch==N && im_channels==channels && im_H==H_IN && im_W == W_IN ) ;
    CHECK_DIMS(offset_batch==N && offset_channels== num_deformable_group*kernel*kernel*2
     && offset_H==H_OUT && offset_W == W_OUT ) ;
    CHECK_DIMS(mask_batch==N && mask_channels== num_deformable_group*kernel*kernel
     && mask_H==H_OUT && mask_W == W_OUT ) ;
    CHECK_DIMS(weight_outchannels == filters && weight_inchannels== channels*kernel*kernel) ;
    if(use_bias){
        CHECK_DIMS(bias_outchannels == filters );
    }

    // im2col
    int64_t col4d_shape_list[4] = {channels*kernel*kernel,N,H_OUT,W_OUT} ;
    at::IntList col4d_shape(col4d_shape_list,4) ;
    at::TensorOptions options= weights.options();
    options.requires_grad(false);
    //options.is_variable(true);
    options.device(weights.device());
    options.dtype(weights.dtype());
    at::Tensor col4d = at::empty(col4d_shape,options);
    int64_t kernel_shape_list[2] = {kernel,kernel};
    at::IntList kernel_shape(kernel_shape_list,2) ;
    int64_t pad_shape_list[2] = {pad,pad};
    at::IntList pad_shape(pad_shape_list,2) ;
    int64_t stride_shape_list[2] = {stride,stride};
    at::IntList stride_shape(stride_shape_list,2) ;
    int64_t dilation_shape_list[2] = {dilation,dilation};
    at::IntList dilation_shape(dilation_shape_list,2) ;

    CHECK_INPUT(im);
    CHECK_INPUT(Offset);
    CHECK_INPUT(Mask);
    modulated_deformable_im2col(
                                im, Offset, Mask,
                                im_shape, col4d_shape,
                                kernel_shape,pad_shape,stride_shape,dilation_shape,
                                num_deformable_group,
                                col4d);
    // col2out
    int64_t col2d_shape_list[2]={col4d_shape[0],N*H_OUT*W_OUT};
    at::IntList col2d_reshape(col2d_shape_list,2) ;
    at::Tensor col2d = col4d.reshape(col2d_reshape); // C*kernel*kernel,N*H_OUT*W_OUT
    at::Tensor out = at::mm(weights,col2d); // C`,N*H_OUT*W_OUT
    //col.reshape(col_shape); //reshape back to 4D

    int64_t bias2d_shape[2]={filters,1};
    at::IntList bias2d_reshape(bias2d_shape,2);
    if (use_bias){
        out = at::add(out,bias.reshape(bias2d_reshape));
        bias.reshape(bias_shape); //reshape back to 1D
    }

    int64_t out4d_shape_list[4]={filters,N,H_OUT,W_OUT};
    at::IntList out4d_shape(out4d_shape_list,4) ;
    out = out.reshape(out4d_shape); // C`,N,H_OUT,W_OUT

    int64_t out4d_permute_list[4]={1,0,2,3};
    at::IntList out4d_permute_shape(out4d_permute_list,4) ;
    out = out.permute(out4d_permute_shape).contiguous() ;
    CHECK_INPUT(out); // N,C`,H_OUT,W_OUT

    std::vector<at::Tensor> output({out,col2d});
    return output ;
}

std::vector<at::Tensor> dcnv2_backward(
        at::Tensor grad_out,
        at::Tensor im,
        at::Tensor weights,
        at::Tensor Offset,
        at::Tensor Mask,
        at::Tensor col,
        int64_t num_deformable_group,
        int64_t kernel,
        int64_t pad,
        int64_t stride,
        int64_t dilation,
        bool use_bias,
        at::Tensor bias
        ) {
        /*

        :param grad_out : N, C`, H, W
        :param im: N,C,H,W
        :param weights: C`, C*kernel*kernel
        :param Offset: N,num_deformable_group*kernel*kernel*2,H,W
        :param Mask: N,num_deformable_group*kernel*kernel,H,W
        :param col : C*kernel*kernel,NHW
        :param num_deformable_group: int , default 1
        :param kernel: int64_t
        :param pad: int64_t
        :param stride: int64_t
        :param dilation: int64_t
        :param use_bias : use bias or not
        :param bias: C` or None
        :return:
            std::vector<at::Tensor> [grad_im,grad_weights,grad_bias,grad_offset,grad_mask]
                                        or [grad_im,grad_weights,grad_offset,grad_mask]
        */
    // maybe not need check contiguous
    CHECK_INPUT(grad_out);
    CHECK_INPUT(weights);
    if(use_bias){
        CHECK_INPUT(bias);
    }
    CHECK_INPUT(col);

    // CHECK Tensor Shape
    // Now we only support 3×3,stride=1,pad=1 conv
    //CHECK_DIMS (kernel == 3) ;
    //CHECK_DIMS (pad == 1 ) ;
    //CHECK_DIMS (stride == 1) ;
    //CHECK_DIMS (dilation == 1) ;

    CHECK_DIMS(im.dim()==4);
    CHECK_DIMS(weights.dim()==2);
    if(use_bias){
        CHECK_DIMS(bias.dim()==1);
    }
    CHECK_DIMS(Offset.dim()==4);
    CHECK_DIMS(Mask.dim()==4);
    CHECK_DIMS(col.dim()==2);
    CHECK_DIMS(grad_out.dim()==4);

    at::IntList grad_out_shape = grad_out.sizes();
    int64_t grad_out_batch = grad_out_shape[0];
    int64_t grad_out_channels = grad_out_shape[1];
    int64_t grad_out_H = grad_out_shape[2];
    int64_t grad_out_W = grad_out_shape[3];

    at::IntList im_shape = im.sizes();
    int64_t im_batch = im_shape[0];
    int64_t im_channels = im_shape[1];
    int64_t im_H = im_shape[2];
    int64_t im_W = im_shape[3];

    at::IntList offset_shape = Offset.sizes();
    int64_t offset_batch = offset_shape[0];
    int64_t offset_channels = offset_shape[1];
    int64_t offset_H = offset_shape[2];
    int64_t offset_W = offset_shape[3];

    at::IntList mask_shape = Mask.sizes();
    int64_t mask_batch = mask_shape[0];
    int64_t mask_channels = mask_shape[1];
    int64_t mask_H = mask_shape[2];
    int64_t mask_W = mask_shape[3];

    at::IntList col_shape = col.sizes();
    int64_t col_channels = col_shape[0];
    int64_t col_NHW = col_shape[1];

    at::IntList weight_shape = weights.sizes();
    int64_t weight_outchannels = weight_shape[0];
    int64_t weight_inchannels = weight_shape[1];

    at::IntList bias_shape;
    int64_t bias_outchannels;
    if(use_bias) {
        bias_shape = bias.sizes();
      bias_outchannels = bias_shape[0];
    }

    const int64_t N = im_batch ;
    const int64_t channels = im_channels ;
    const int64_t filters = weight_outchannels ;
    const int64_t H_IN = im_H ;
    const int64_t W_IN = im_W ;
    const int64_t H_OUT = (H_IN+2*pad-((kernel-1)*dilation+1))/stride + 1 ;
    const int64_t W_OUT = (W_IN+2*pad-((kernel-1)*dilation+1))/stride + 1 ;

    CHECK_DIMS(grad_out_batch==N  && grad_out_channels == filters && grad_out_H==H_OUT && grad_out_W == W_OUT ) ;
    CHECK_DIMS(im_batch==N && im_channels== channels && im_H==H_IN && im_W == W_IN ) ;
    CHECK_DIMS(offset_batch==N && offset_channels== num_deformable_group*kernel*kernel*2
     && offset_H==H_OUT && offset_W == W_OUT ) ;
    CHECK_DIMS( mask_batch==N && mask_channels== num_deformable_group*kernel*kernel
     && mask_H==H_OUT && mask_W == W_OUT ) ;
    CHECK_DIMS(col_channels == channels * kernel * kernel && col_NHW == N*H_OUT*W_OUT ) ;
    CHECK_DIMS(weight_outchannels == filters && weight_inchannels== channels*kernel*kernel) ;
    if (use_bias){
        CHECK_DIMS( bias_outchannels == filters );
    }

    // // gradient w.r.t. input coordinate col
    int64_t gradout4d_permute_list[4]={1,0,2,3}; // NC`HW -> C`NHW
    at::IntList gradout4d_permute_shape(gradout4d_permute_list,4) ;
    at::Tensor gradout_permuted4d = grad_out.permute(gradout4d_permute_shape); //.contiguous() ;

    int64_t gradout2d_shape_list[2]={filters,N*H_OUT*W_OUT}; // C`NHW -> C`,NHW
    at::IntList gradout2d_shape(gradout2d_shape_list,2) ;
    at::Tensor gradout_permuted2d  = gradout_permuted4d.reshape(gradout2d_shape);

    int64_t weight2d_permute_list[2]={1,0}; // NC`HW -> C`NHW
    at::IntList weight2d_permute_shape(weight2d_permute_list,2) ;
    at::Tensor weights_tranposed2d = weights.permute(weight2d_permute_shape); // C`,C*kernel*kernel -> C*kernel*kernel,C`

    at::Tensor grad_col2d = at::mm(weights_tranposed2d,gradout_permuted2d); // C*kernel*kernel , NHW
    //at::IntList gradcol2d_shape = grad_col2d.sizes();

    // gradient w.r.t. input coordinate data
    at::TensorOptions options= weights.options();
    options.requires_grad(false);
    //options.is_variable(true);
    options.device(weights.device());
    options.dtype(weights.dtype());
    at::Tensor grad_offset = at::zeros(offset_shape,options);
    at::Tensor grad_mask = at::zeros(mask_shape,options);

    int64_t kernel_list[2] = {kernel,kernel};
    at::IntList kernel_shape(kernel_list,2) ;
    int64_t pad_list[2] = {pad,pad};
    at::IntList pad_shape(pad_list,2) ;
    int64_t stride_list[2] = {stride,stride};
    at::IntList stride_shape(stride_list,2) ;
    int64_t dilation_list[2] = {dilation,dilation};
    at::IntList dilation_shape(dilation_list,2) ;

    int64_t grad_col4d_list[4] = {channels*kernel*kernel,N,H_OUT,W_OUT} ;  // C*kernel*kernel , N , H , W
    at::IntList grad_col4d_shape(grad_col4d_list,4);
    at::Tensor grad_col4d = grad_col2d.reshape(grad_col4d_shape);

    CHECK_INPUT(grad_col4d);
    CHECK_INPUT(im);
    CHECK_INPUT(Offset);
    CHECK_INPUT(Mask);
    CHECK_INPUT(grad_offset);
    CHECK_INPUT(grad_mask);
    // gradient w.r.t. im coord
    modulated_deformable_col2im_coord(
                                      grad_col4d, im, Offset, Mask,
                                      im_shape, grad_col4d_shape,
                                      kernel_shape,pad_shape,stride_shape,dilation_shape,
                                      num_deformable_group,
                                      grad_offset,grad_mask);
    // gradient w.r.t. im data
    at::Tensor grad_im = at::zeros(im_shape,options);
    CHECK_INPUT(grad_col4d);
    CHECK_INPUT(Offset);
    CHECK_INPUT(Mask);
    CHECK_INPUT(grad_im);
    modulated_deformable_col2im(
                                grad_col4d, Offset, Mask,
                                im_shape, grad_col4d_shape,
                                kernel_shape,pad_shape,stride_shape,dilation_shape,
                                num_deformable_group,
                                grad_im);

    // gradient w.r.t. weight, dWeight should accumulate across the batch and group
    // save time :try use col data in forward ? Done
    // save meomory : try cover grad_col ? Done
    // modulated_deformable_im2col<at::Tensor>( // forward again
    //                                    im, Offset, Mask,
    //                                    im_shape, gradcol_shape,
    //                                    kernel,pad,stride,dilation,
    //                                    num_deformable_group,
    //                                    grad_col);
    // at::Tensor col_tranpose = grad_col.transpose(0,1);   // 9C , NH`W` -> NH`W` ,9C
    int64_t col_permute2d_list[2] = {1,0} ;
    at::IntList col_permute2d_shape(col_permute2d_list,2);
    at::Tensor col_permute2d = col.permute(col_permute2d_shape);// NH`W`,C*kernel*kernel ,
    // need averaged by NHW ?
    at::Tensor grad_weights = at::mm(gradout_permuted2d,col_permute2d); // C`,C*kernel*kernel

    if(use_bias){
        // gradient w.r.t bias
        // need averaged by NHW ?
        int64_t sumdim_shape[1] = {1};
        at::IntList sumdim(sumdim_shape,1);
        at::Tensor grad_bias = at::sum(grad_out,sumdim); // C`
        std::vector<at::Tensor> out({grad_im,grad_weights,grad_offset,grad_mask,grad_bias});
        return out;
    }
    else {
        std::vector<at::Tensor> out({grad_im,grad_weights,grad_offset,grad_mask});
        return out;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dcnv2_forward, "Deformable ConvNets v2 forward (CUDA)");
  m.def("backward", &dcnv2_backward, "Deformable ConvNets v2 backward (CUDA)");
}
