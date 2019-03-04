/* Convolution of 3 channels horizontally in an optimized way for a separable convolution and output w/h flipping
__kernel void convolve3Channels(    const __global float3* input,
                                    __constant float3* kernel,
                                    __global float3* output)
{
    const int pixel = get_global_id(0);


}
// Convolution of a specific channel horizontally for a separable convolution and output w/h flipping
__kernel void convolve1Channel(    const __global float3* input,
                                    __constant float3* kernel,
                                    __global float3* output)
{
    const int pixel = get_global_id(0);


}*/

//We have to use float4 instead of float3 because of compatibility with OpenCL 1.0 and alignment issues
__constant float4 RGB2XYZm[3] = {(float4)(0.4124564f, 0.3575761f, 0.1804375f,0.0f),(float4)(0.2126729f, 0.7151522f, 0.0721750f,0.0f),(float4)(0.0193339f, 0.1191920f, 0.9503041f,0.0f)};

__kernel void RGB2XYZ(     __global float* inputR,
                            __global float* inputG,
                            __global float* inputB,
                            __global float4* output)
{
    const int pixel = get_global_id(0);
    const float R = (inputR[pixel] <= 0.04045f ? inputR[pixel]/12.92f : pow((inputR[pixel]+0.055f)/1.055f,2.4f));
    const float G = (inputG[pixel] <= 0.04045f ? inputG[pixel]/12.92f : pow((inputG[pixel]+0.055f)/1.055f,2.4f));
    const float B = (inputB[pixel] <= 0.04045f ? inputB[pixel]/12.92f : pow((inputB[pixel]+0.055f)/1.055f,2.4f));
    const float4 RGB = (float4)(R,G,B,0.0f);
    output[pixel] = (float4)(dot(RGB, RGB2XYZm[0]),dot(RGB, RGB2XYZm[1]),dot(RGB, RGB2XYZm[2]),0.0f);
}

__constant float4 XYZ2RGBm[3] = {(float4)(3.2404542f, -1.5371385f, -0.4985314f,0.0f),(float4)(-0.9692660f, 1.8760108f, 0.0415560f,0.0f),(float4)(0.0556434f, -0.2040259f, 1.0572252f,0.0f)};

__kernel void XYZ2RGB(  __global float4* inputXYZ,
                        __global float* Rout,
                        __global float* Gout,
                        __global float* Bout
                        )
{
    const int pixel = get_global_id(0);
    const float inv = 1.0f/2.4f;
    const float R = dot(inputXYZ[pixel], XYZ2RGBm[0]);
    const float G = dot(inputXYZ[pixel], XYZ2RGBm[1]);
    const float B = dot(inputXYZ[pixel], XYZ2RGBm[2]);
    Rout[pixel] = (R <= 0.0031308f ? R*12.92f : pow(R*1.055f, inv)-0.055f);
    Gout[pixel] = (G <= 0.0031308f ? G*12.92f : pow(G*1.055f, inv)-0.055f);
    Bout[pixel] = (B <= 0.0031308f ? B*12.92f : pow(B*1.055f, inv)-0.055f);
}