//Convolution of 4 channels horizontally in an optimized way for a separable convolution and output w/h flipping
__kernel void convolve4Channels(    const __global float4* input,
                                    __constant float4* k,
                                    int halfSize,
                                    int imageW,
                                    int imageH,
                                    int update,
                                    __global float4* output)
{
    const int pixel = get_global_id(0);
    // pixel = i*w +j
    const int j = pixel%imageW;
    const int linestart = (pixel/imageW)*imageW;
    const int outPixel = j*imageH + (pixel/imageW);
    float4 temp = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    int off;
    int kOff=0;
    for(int i=-halfSize; i <= halfSize; i++, kOff++)
    {
        off = j+i;
        if(off < 0)
        {
            off = -off-1;
        }else if(off >= imageW)
        {
            off = (imageW << 1)-off-1;
        }
        temp=fma(input[linestart+off],k[kOff],temp);
    }
    if(update == 1)
    {
        output[outPixel]+=temp;
    }else
    {
        output[outPixel]=temp;
    }



}
//Convolution of a specific channel horizontally for a separable convolution and output w/h flipping
__kernel void convolve1Channel(    const __global float4* input,
                                   __constant float4* k,
                                   int halfSize,
                                   int imageW,
                                   int imageH,
                                   int update,
                                   __global float4* output)
{
    const int pixel = get_global_id(0);
    // pixel = i*w +j
    const int j = pixel%imageW;
    const int linestart = (pixel/imageW)*imageW;
    const int outPixel = j*imageH + (pixel/imageW);
    float4 temp = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    int off;
    int kOff=0;
    for(int i=-halfSize; i <= halfSize; i++)
    {
        off = j+i;
        if(off < 0)
            off = -off-1;
        if(off >= imageW)
            off = (imageW << 1)-off-1;
        temp.x=fma(input[linestart+off].x,k[kOff++].x,temp.x);
    }
    if(update == 1)
    {
        output[outPixel].x+=temp.x;
    }else
    {
        output[outPixel].x=temp.x;
    }
}

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
    Rout[pixel] = (R <= 0.0031308f ? R*12.92f : 1.055f*pow(R, inv)-0.055f);
    Gout[pixel] = (G <= 0.0031308f ? G*12.92f : 1.055f*pow(G, inv)-0.055f);
    Bout[pixel] = (B <= 0.0031308f ? B*12.92f : 1.055f*pow(B, inv)-0.055f);
}

__constant float4 XYZ2Oppm[3] = {(float4)(0.2787336f, 0.7218031f, -0.1065520f, 0.0f),(float4)(-0.4487736f,0.2898056f,-0.0771569f, 0.0f),(float4)(0.0859513f,-0.5899859f,0.5011089f, 0.0f)};
__kernel void XYZ2Opp(  __global float4* inputXYZ,
                        __global float4* output)
{
    const int pixel = get_global_id(0);
    output[pixel] = (float4)(dot(inputXYZ[pixel], XYZ2Oppm[0]),dot(inputXYZ[pixel], XYZ2Oppm[1]),dot(inputXYZ[pixel], XYZ2Oppm[2]), 0.0f);
}

__constant float4 Opp2XYZm[3] = {(float4)(0.624045f, -1.87044f, -0.155304f, 0.0f),(float4)(1.36606f, 0.931563f, 0.433903f, 0.0f),(float4)(1.5013f, 1.41761f, 2.53307f, 0.0f)};
__constant float LABDELTA = 6.0f/29.0f;
__constant float LABDELTA2 = 36.0f/841.0f;
__constant float LABDELTA3= 216.0f/24389.0f;
__kernel void Opp2LAB(  __global float4* inputOpp,
                        float illuminantX,
                        float illuminantY,
                        float illuminantZ,
                        __global float4* output)
{
    const int pixel = get_global_id(0);
    const float X = dot(inputOpp[pixel], Opp2XYZm[0]);
    const float Y = dot(inputOpp[pixel], Opp2XYZm[1]);
    const float Z = dot(inputOpp[pixel], Opp2XYZm[2]);

    float t = X/illuminantX;
    const float fx = (t > LABDELTA3) ? pow(t, 1.0f / 3.0f) : ((t / (3 * LABDELTA2)) + (4.0f / 29.0f));
    t = Y/illuminantY;
    const float fy = (t > LABDELTA3) ? pow(t, 1.0f / 3.0f) : ((t / (3 * LABDELTA2)) + (4.0f / 29.0f));
    t = Z/illuminantZ;
    const float fz = (t > LABDELTA3) ? pow(t, 1.0f / 3.0f) : ((t / (3 * LABDELTA2)) + (4.0f / 29.0f));
    output[pixel] = (float4)(116.0f*fy-16.0f,500.0f*(fx-fy),200.0f*(fy-fz), 0.0f);
}
