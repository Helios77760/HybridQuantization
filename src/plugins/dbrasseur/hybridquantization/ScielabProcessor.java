package plugins.dbrasseur.hybridquantization;

import icy.type.collection.array.Array1DUtil;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;


/**
 * Class for S-cielab processing
 * S-CIELAB values and code are derived from the S-CIELAB example code http://scarlet.stanford.edu/~brian/scielab/scielab.html
 * XYZ-Opp conversion matrices are from A HYBRID COLOR QUANTIZATION ALGORITHM INCORPORATING A HUMAN VISUAL PERCEPTION MODEL
 */
public class ScielabProcessor {
	public enum Whitepoint{D50, D65}
	private static final float[] D65 = {0.95047f, 1.0f, 1.0883f};
	private static final float[] D50 = {0.966797f, 1.0f, 0.825188f};

	private static final int minSAMPPERDEG = 224;   //As specified in the S-CIELAB example code
	public static final float[][] mSRGBtoXYZ = {
			{0.4124564f, 0.3575761f, 0.1804375f},
			{0.2126729f, 0.7151522f, 0.0721750f},
			{0.0193339f, 0.1191920f, 0.9503041f}
	};
	public static final float[][] mXYZtoSRGB = {
			{3.2404542f, -1.5371385f, -0.4985314f},
			{-0.9692660f, 1.8760108f, 0.0415560f},
			{0.0556434f, -0.2040259f, 1.0572252f}
	};
	public static final float[][] mXYZtoOpp = {
			{0.2787336f, 0.7218031f, -0.1065520f},
			{-0.4487736f,0.2898056f,0.0771569f},
			{0.0859513f,-0.5899859f,0.5011089f}
	};
	public static final float[][] mOpptoXYZ = {
			{0.97959616044562807864f, -1.5347157012664408981f, 0.44459764330437399288f},
			{1.188977906742323787f, 0.7643549575179937615f, 0.13512574791125839373f},
			{1.2318333139247290457f, 1.1631592597636512884f, 2.0784075888008567862f}
	};
	private static final float[][] weigths={
			{1.00327f,0.114416f, -0.117686f},
			{0.616725f, 0.383275f},
			{0.567885f, 0.432115f}
	};
	private static final float[][] halfwidths={
			{0.05f,0.225f,7.0f},
			{0.0685f,0.826f},
			{0.0920f,0.6451f}
	};
	private float[][][] Ofilters;
	private float[][][] absOfilters;
	private float[] illuminant;
	private ImageManipulation imageProcessing;

	private float[][] float4filters;
	private float[][] absFloat4filters;
	private boolean openCLfiltersPrepared;

	private static final float LABDELTA = 6.0f/29.0f;
	private static final float LABDELTA2 = LABDELTA*LABDELTA;
	private static final float LABDELTA3= LABDELTA2*LABDELTA;


	public ScielabProcessor(int dpi, double viewingDistance, Whitepoint whitepoint)
	{
		//Set the whitepoint
		if(whitepoint == Whitepoint.D50)
		{
			illuminant = Arrays.copyOf(D50, D50.length);
		}else
		{
			illuminant = Arrays.copyOf(D65, D65.length);
		}

		//Calculate the uprate for filter creation :
		int uprate;
		int sampPerDeg = (int)Math.round(dpi /((180/Math.PI)*Math.atan(2.54/viewingDistance)));
		if(sampPerDeg < minSAMPPERDEG)
		{
			uprate = (int)Math.ceil(minSAMPPERDEG*1.0/sampPerDeg);
			sampPerDeg *= uprate;
		}else
		{
			uprate = 1;
		}

		//We convert the halfwidths from visual angle to pixels (by multiplying all of them by the number of samples per degree)
		float[][] spreads = new float[3][];
		for(int i = 0; i < halfwidths.length; i++)
		{
			spreads[i] = new float[halfwidths[i].length];
			for(int j=0; j<halfwidths[i].length; j++)
			{
				spreads[i][j] = halfwidths[i][j]*sampPerDeg;
			}
		}

		//We limit the width of the filters to 1 degree of visual angle and to a odd number of points
		int width = (int)(Math.ceil(sampPerDeg/2.0))*2-1;
		//Generating the separable filters
		Ofilters = new float[3][][];
		Ofilters[0] = new float[3][];
		Ofilters[1] = new float[2][];
		Ofilters[2] = new float[2][];
		for(int i =0; i < Ofilters.length; i++)
		{
			for(int j=0; j<Ofilters[i].length; j++)
			{
				Ofilters[i][j] = gauss(spreads[i][j], width);
				float factor = (float)Math.sqrt(Math.abs(weigths[i][j]))* Math.signum(weigths[i][j]);
				for(int k=0; k<Ofilters[i][j].length; k++)
				{
					Ofilters[i][j][k]*=factor;
				}
			}
		}

		//Upsampling and downsampling
		if(uprate > 1)
		{
			//HybridQuantization.addPerfLabel(HybridQuantization.perfTime, "test");
			//Generate upsampling kernel
			float[] upcol = new float[uprate*2-1];
			for(int i=0; i<upcol.length; i++)
			{
				upcol[i] = (uprate - Math.abs(uprate - i-1))*1.0f/uprate;
			}
			//Resize it
			upcol = resize1D(upcol, upcol.length+width-1);
			//Convolve it with the opponent filters
			float[][][] ups = new float[3][][];
			ups[0]= new float[3][];
			ups[0][0] = conv1D(Ofilters[0][0], upcol);
			ups[0][1] = conv1D(Ofilters[0][1], upcol);
			ups[0][2] = conv1D(Ofilters[0][2], upcol);
			ups[1]= new float[2][];
			ups[1][0] = conv1D(Ofilters[1][0], upcol);
			ups[1][1] = conv1D(Ofilters[1][1], upcol);
			ups[2]= new float[2][];
			ups[2][0] = conv1D(Ofilters[2][0], upcol);
			ups[2][1] = conv1D(Ofilters[2][1], upcol);
			//Generate the downsampling indices
			int s = ups[0][0].length;
			int mid = s/2;
			int[] downs = new int[2*(mid/uprate)+1];
			List<Integer> temp = IntStream.iterate(mid, i -> i-uprate) // next int
					.limit((mid/uprate)+1) // only numbers in range
					.boxed()
					.collect(toList());
			Collections.reverse(temp);
			for(int i=0, j=mid+uprate; i<downs.length;i++)
			{
				if(temp.size() > i)
					downs[i] = temp.get(i);
				else
				{
					downs[i] = j;
					j += uprate;
				}

			}
			//Downsample the kernels
			Ofilters[0][0] = extractWithIndices(ups[0][0], downs);
			Ofilters[0][1] = extractWithIndices(ups[0][1], downs);
			Ofilters[0][2] = extractWithIndices(ups[0][2], downs);
			Ofilters[1][0] = extractWithIndices(ups[1][0], downs);
			Ofilters[1][1] = extractWithIndices(ups[1][1], downs);
			Ofilters[2][0] = extractWithIndices(ups[2][0], downs);
			Ofilters[2][1] = extractWithIndices(ups[2][1], downs);
		}
		absOfilters = new float[3][][];
		absOfilters[0]= new float[3][];
		absOfilters[0][0] = new float[Ofilters[0][0].length];
		absOfilters[0][1] = new float[Ofilters[0][1].length];
		absOfilters[0][2] = new float[Ofilters[0][2].length];
		absOfilters[1]= new float[2][];
		absOfilters[1][0] = new float[Ofilters[1][0].length];
		absOfilters[1][1] = new float[Ofilters[1][1].length];
		absOfilters[2]= new float[2][];
		absOfilters[2][0] = new float[Ofilters[2][0].length];
		absOfilters[2][1] = new float[Ofilters[2][1].length];
		for(int i=0; i<absOfilters.length; i++)
		{
			for(int j=0; j<absOfilters[i].length;j++)
			{
				for(int k=0; k<absOfilters[i][j].length; k++)
				{
					float val = Ofilters[i][j][k];
					absOfilters[i][j][k] = val < 0.0f ? -val : val;
				}
			}
		}
		imageProcessing = new ImageManipulation();
		openCLfiltersPrepared =false;
	}



	private static float[] conv1D(float[] data, float[] filter)
	{
		float[] result = new float[data.length];
		Array1DUtil.fill(result, 0.0);
		int offset = filter.length/2;
		for(int i=0; i<data.length;i++)
		{
			for(int j=-offset; j <= offset;j++)
			{
				if(!(i-j < 0 || i+j >= data.length))
				{
					result[i]+=filter[j+offset]*data[i];
				}
			}
		}
		return result;
	}

	private static float[] resize1D(float[] src, int newSize)
	{
		float[] res = new float[newSize];
		int pad = Math.abs(newSize-src.length)/2;
		int j=0;
		if(newSize > src.length)
		{
			for(int i=0; i<newSize; i++)
			{
				res[i] = i < pad || i >= pad+src.length ? 0 : src[j++];
			}
			return res;
		}else
		{
			System.arraycopy(src, pad, res, 0, newSize);
			return res;
		}
	}

	private static float[] extractWithIndices(float[] src, int[] indices)
	{
		float[] result = new float[indices.length];
		for(int i=0; i<indices.length;i++)
		{
			result[i] = src[indices[i]];
		}
		return result;
	}

	/**
	 * Returns a centered gaussian that sums to one
	 * @param halfwidth halfwidth of the gaussian, must be > 1
	 * @param width number of sample points
	 * @return the genrated gaussian, sums to 1
	 */
	public static float[] gauss(float halfwidth, int width)
	{
		float alpha = 2*(float)Math.sqrt(Math.log(2))/(halfwidth-1);
		float[] result = new float[width];
		int offset = width/2;
		double sum=0;
		for(int i=0; i<width;i++)
		{
			result[i] = (float)Math.exp(-alpha*alpha*(i-offset)*(i-offset));
			sum += result[i];
		}
		for(int i=0; i<width;i++)
		{
			result[i]/=sum;
		}
		return result;
	}

	/**
	 * Converts a color to another space linearly by 3*3 matrix multiplication
	 * @param C source color
	 * @param M multiplcation matrix
	 * @return color converted
	 */
	public static float[] matrixColorConvert(float[] C, float[][] M)
	{
		return new float[]{
				C[0]*M[0][0] + C[1]*M[0][1] + C[2]*M[0][2],
				C[0]*M[1][0] + C[1]*M[1][1] + C[2]*M[1][2],
				C[0]*M[2][0] + C[1]*M[2][1] + C[2]*M[2][2],
		};
	}

	public static float[] sRGBtoXYZ(float[] sRGB)
	{
		float[] RGB = {sRGB[0] <= 0.04045f ? sRGB[0]/12.92f : (float)Math.pow((sRGB[0]+0.055f)/1.055f,2.4f),
				sRGB[1] <= 0.04045f ? sRGB[1]/12.92f : (float)Math.pow((sRGB[1]+0.055f)/1.055f,2.4f),
				sRGB[2] <= 0.04045f ? sRGB[2]/12.92f : (float)Math.pow((sRGB[2]+0.055f)/1.055f,2.4f)};
		return matrixColorConvert(RGB, mSRGBtoXYZ);
	}

	public float[] sRGBtoOpp(float[] RGB)
	{
		//Gamma correction
		float R = (RGB[0] <= 0.04045f) ? (RGB[0] / 12.92f) : (float)Math.pow((RGB[0] + 0.055f) / 1.055f, 2.4f);
		float G = (RGB[1] <= 0.04045f) ? (RGB[1] / 12.92f) : (float)Math.pow((RGB[1] + 0.055f) / 1.055f, 2.4f);
		float B = (RGB[2] <= 0.04045f) ? (RGB[2] / 12.92f) : (float)Math.pow((RGB[2] + 0.055f) / 1.055f, 2.4f);
		//Matrix multiplication XYZ2OPP*RGB2XYZ*RGB
		return new float[]{
				0.26641335000823f*R + 0.60316740257478f*G + 0.0011333302293f*B,
				-0.12197400229389f*R+ 0.05598088396616f*G + 0.01326365114329f*B,
				-0.08033445917708f*R+ -0.33146741170125f*G + 0.44913244757774f*B
		};
	}

	public float[] OpptosLab(float[] Opp)
	{
		//Matrix multiplication Opp2XYZ*Opp
		float X = 0.97959616044562807864f*Opp[0] + -1.5347157012664408981f*Opp[1] + 0.44459764330437399288f*Opp[2];
		float Y = 1.188977906742323787f*Opp[0] + 0.7643549575179937615f*Opp[1] + 0.13512574791125839373f*Opp[2];
		float Z = 1.2318333139247290457f*Opp[0] + 1.1631592597636512884f*Opp[1] + 2.0784075888008567862f*Opp[2];
		//XYZtoLAB
		float t = X/illuminant[0];
		float fx = (t > LABDELTA3) ? (float)Math.pow(t, 1.0 / 3.0) : ((t / (3 * LABDELTA2)) + (4.0f / 29.0f));
		t = Y/illuminant[1];
		float fy = (t > LABDELTA3) ? (float)Math.pow(t, 1.0 / 3.0) : ((t / (3 * LABDELTA2)) + (4.0f / 29.0f));
		t = Z/illuminant[2];
		float fz = (t > LABDELTA3) ? (float)Math.pow(t, 1.0 / 3.0) : ((t / (3 * LABDELTA2)) + (4.0f / 29.0f));
		return new float[]{
				116.0f*fy-16.0f,
				500.0f*(fx-fy),
				200.0f*(fy-fz)
		};
	}

	public static float[] XYZtosRGB(float[] XYZ)
	{
		float[] RGB = matrixColorConvert(XYZ, mXYZtoSRGB);
		return new float[]{
				RGB[0] <= 0.0031308f ? RGB[0]*12.92f : (float)Math.pow(RGB[0]*1.055f, 1.0f/2.4f)-0.055f,
				RGB[1] <= 0.0031308f ? RGB[1]*12.92f : (float)Math.pow(RGB[1]*1.055f, 1.0f/2.4f)-0.055f,
				RGB[2] <= 0.0031308f ? RGB[2]*12.92f : (float)Math.pow(RGB[2]*1.055f, 1.0f/2.4f)-0.055f
		};
	}

	public static float[] XYZtoOpp(float[] XYZ)
	{
		return matrixColorConvert(XYZ, mXYZtoOpp);
	}

	public static float[] OppToXYZ(float[] Opp)
	{
		return matrixColorConvert(Opp, mOpptoXYZ);
	}

	private float[] XYZtoLAB(float[] XYZ)
	{
		float fx = LAB_f(XYZ[0]/ illuminant[0]);
		float fy = LAB_f(XYZ[1]/ illuminant[1]);
		float fz = LAB_f(XYZ[2]/ illuminant[2]);
		return new float[]{
				116.0f*fy-16.0f,
				500.0f*(fx-fy),
				200.0f*(fy-fz)
		};
	}

	private float[] LABtoXYZ(float[] LAB)
	{
		float L = (LAB[0]+16.0f)/116.0f;
		return new float[]
				{
						illuminant[0]*LAB_finv(L + LAB[1]/500.0f),
						illuminant[1]*LAB_finv(L),
						illuminant[2]*LAB_finv(L - LAB[2]/200.0f)
				};
	}

	private static float LAB_f(float t)
	{
		float delta = 6.0f/29.0f;
		return (t > delta*delta*delta ? (float)Math.pow(t, 1.0/3.0) : t/(3*delta*delta) + 4.0f/29.0f);
	}

	private static float LAB_finv(float t)
	{
		float delta = 6.0f/29.0f;
		return t > delta ? t*t*t : 3*delta*delta*(t-4.0f/29.0f);
	}

	/**
	 * Converts an sRBG image to its S-CIELAB representation
	 * @param image sRGB image in an [C][XY] array
	 * @param w width of the image in pixels
	 * @return Lab image in an [C][XY] array in its S-CIELAB representation
	 */
	public float[][] imageToScielab(float[][] image, int w)
	{
		//First we convert the RGBImage to XYZ
		float[] inlineImageXYZ = imageProcessing.RGBtoXYZ(image[0], image[1], image[2]);
		//Then we convert to Poirson&Wandell opponent and apply the filters for S-CIELAB, then convert to CIELAB
		float[] ImageLAB = imageProcessing.XYZtoScielab(inlineImageXYZ, Ofilters,absOfilters, w, illuminant);

		//float[] inlineImage2 = new float[4*image[0].length];
        /*for(int i=0; i<image[0].length; i++)
        {
            float[] XYZ =ScielabProcessor.sRGBtoXYZ(new float[]{image[0][i], image[1][i], image[2][i]});
            int off = i<<2;
            inlineImage2[off] = XYZ[0];
            inlineImage2[off+1] = XYZ[1];
            inlineImage2[off+2] = XYZ[2];
            inlineImage2[off+3] = 0.0f;
        }*/

        /*for(int i=0; i<inlineImageXYZ.length/4;i++)
        {
            int off = i <<2;
            result[0][i] = inlineImageXYZ[off];
            result[1][i] = inlineImageXYZ[off+1];
            result[2][i] = inlineImageXYZ[off+2];
        }*/
		/*
		//First we convert to XYZ and then to Poirson&Wandell opponent

		float[][] XYC = new float[result[0].length][3];
		long time = System.currentTimeMillis();
		for(int i=0; i<XYC.length;i++)
		{
			XYC[i][0]=image[0][i];
			XYC[i][1]=image[1][i];
			XYC[i][2]=image[2][i];
		}

		Arrays.parallelSetAll(XYC, i->sRGBtoOpp(XYC[i]));

		//-> Converting back to [C][XY]
		/*for(int i=0; i<XYC.length;i++)
		{
			result[0][i]=XYC[i][0];
			result[1][i]=XYC[i][1];
			result[2][i]=XYC[i][2];
		}*/
		/*
		float[] resultInline = new float[XYC.length*4]; //La specification OpenCL impose un alignement de 4*sizeof(float)
		int off;
		for(int i=0; i< XYC.length; i++)
        {
            off = i*3;
            resultInline[off]=XYC[i][0];
            resultInline[off+1]=XYC[i][1];
            resultInline[off+2]=XYC[i][2];
            resultInline[off+3]=0.0f;
        }*/

		//HybridQuantization.perfTime = HybridQuantization.addPerfLabel(time, "RGB2Opp");

		/*for(int x=0; x<h; x++)
		{
			for(int y=0; y<w; y++)
			{
				offset = x*w + y;
				srcBufferPixel[0] = image[0][offset];
				srcBufferPixel[1] = image[1][offset];
				srcBufferPixel[2] = image[2][offset];
				outBufferPixel = XYZtoOpp(sRGBtoXYZ(srcBufferPixel));
				result[0][offset] = outBufferPixel[0];
				result[1][offset] = outBufferPixel[1];
				result[2][offset] = outBufferPixel[2];
			}
		}*/
		/*
		long start = System.currentTimeMillis();
		//Then we apply filters to mimic human vision
		imageProcessing.convolve(resultInline, Ofilters, w);
		HybridQuantization.perfTime = HybridQuantization.addPerfLabel(start, "Convolution");

		for(int i=0; i<XYC.length;i++)
		{
			XYC[i][0]=result[0][i];
			XYC[i][1]=result[1][i];
			XYC[i][2]=result[2][i];
		}

		//Switch back to XYZ and then to LAB
		Arrays.parallelSetAll(XYC, i->OpptosLab(XYC[i]));

		//-> Converting back to [C][XY]
		for(int i=0; i<XYC.length;i++)
		{
			result[0][i]=XYC[i][0];
			result[1][i]=XYC[i][1];
			result[2][i]=XYC[i][2];
		}
		//HybridQuantization.perfTime = HybridQuantization.addPerfLabel(HybridQuantization.perfTime, "Opp2LAB");

		/*for(int x=0; x<h; x++)
		{
			for(int y=0; y<w; y++)
			{
				offset = x*w + y;
				srcBufferPixel[0] = result[0][offset];
				srcBufferPixel[1] = result[1][offset];
				srcBufferPixel[2] = result[2][offset];
				outBufferPixel = XYZtoLAB(OppToXYZ(srcBufferPixel));
				result[0][offset] = outBufferPixel[0];
				result[1][offset] = outBufferPixel[1];
				result[2][offset] = outBufferPixel[2];
			}
		}*/
		return inlineLabToRGB(ImageLAB);
	}

	public float[][] LabTosRGB(float[][] image)
	{
		float[][] result = new float[3][image[0].length];
		float[] srcBufferPixel = new float[3];
		float[] outBufferPixel;
		for(int i=0; i < image[0].length; i++)
		{
			srcBufferPixel[0] = image[0][i];
			srcBufferPixel[1] = image[1][i];
			srcBufferPixel[2] = image[2][i];
			outBufferPixel = XYZtosRGB(LABtoXYZ(srcBufferPixel));
			result[0][i] = outBufferPixel[0];
			result[1][i] = outBufferPixel[1];
			result[2][i] = outBufferPixel[2];
		}
		return result;
	}

	public float[][] inlineLabToRGB(float[] lab)
	{
		float[][] result = new float[3][lab.length/4];
		float[] outBufferPixel;
		for(int i=0; i < lab.length/4; i++)
		{
			int off = i << 2;
			outBufferPixel = Arrays.copyOfRange(lab, off, off+4);
			outBufferPixel = XYZtoLAB(sRGBtoXYZ(outBufferPixel));
			result[0][i] = outBufferPixel[0];
			result[1][i] = outBufferPixel[1];
			result[2][i] = outBufferPixel[2];
		}
		return result;
	}

	public float[][] sRGBtoLab(float[][] image)
	{
		float[][] result = new float[3][image[0].length];
		float[] srcBufferPixel = new float[3];
		float[] outBufferPixel;
		for(int i=0; i < image[0].length; i++)
		{
			srcBufferPixel[0] = image[0][i];
			srcBufferPixel[1] = image[1][i];
			srcBufferPixel[2] = image[2][i];
			outBufferPixel = XYZtoLAB(sRGBtoXYZ(srcBufferPixel));
			result[0][i] = outBufferPixel[0];
			result[1][i] = outBufferPixel[1];
			result[2][i] = outBufferPixel[2];
		}
		return result;
	}

	public void close()
	{
		imageProcessing.close();
	}
}
