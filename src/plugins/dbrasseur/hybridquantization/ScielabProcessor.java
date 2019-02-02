package plugins.dbrasseur.hybridquantization;

import icy.image.colorspace.IcyColorSpace;
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
	private static final double[] D65 = {0.95047, 1.0, 1.0883};
	private static final double[] D50 = {0.966797, 1.0, 0.825188};

	private static final int minSAMPPERDEG = 50;   //As specified in the S-CIELAB example code
	public static final double[][] mSRGBtoXYZ = {
			{0.4124564, 0.3575761, 0.1804375},
			{0.2126729, 0.7151522, 0.0721750},
			{0.0193339, 0.1191920, 0.9503041}
	};
	public static final double[][] mXYZtoSRGB = {
			{3.2404542, -1.5371385, -0.4985314},
			{-0.9692660, 1.8760108, 0.0415560},
			{0.0556434, -0.2040259, 1.0572252}
	};
	public static final double[][] mXYZtoOpp = {
			{0.279,0.72,-0.107},
			{-0.449,0.29,-0.077},
			{0.086,-0.59,0.501}
	};
	public static final double[][] mOpptoXYZ = {
			{0.62655, -1.86718, -0.15316},
			{1.36986, 0.93476, 0.43623},
			{1.50565, 1.42132, 2.53602}
	};
	private static final double[][] weigths={
			{1.00327,0.114416, -0.117686},
			{0.616725, 0.383275},
			{0.567885, 0.432115}
	};
	private static final double[][] halfwidths={
			{0.05,0.225,7.0},
			{0.0685,0.33},
			{0.0920,0.6451}
	};
	private double[][][] Ofilters;
	private double[] illuminant;

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
		int sampPerDeg = (int)Math.round(dpi *(180/Math.PI)*Math.atan(2.54/viewingDistance));
		if(sampPerDeg < minSAMPPERDEG)
		{
			uprate = (int)Math.ceil(minSAMPPERDEG/sampPerDeg);
			sampPerDeg *= uprate;
		}else
		{
			uprate = 1;
		}

		//We convert the halfwidths from visual angle to pixels (by multiplying all of them by the number of samples per degree)
		double[][] spreads = new double[3][];
		for(int i = 0; i < halfwidths.length; i++)
		{
			spreads[i] = new double[halfwidths[i].length];
			for(int j=0; j<halfwidths[i].length; j++)
			{
				spreads[i][j] = halfwidths[i][j]*sampPerDeg;
			}
		}

		//We limit the width of the filters to 1 degree of visual angle and to a odd number of points
		int width = (sampPerDeg/2)*2+1;
		//Generating the separable filters
		Ofilters = new double[3][][];
		Ofilters[0] = new double[3][];
		Ofilters[0][0] = Arrays.stream(gauss(spreads[0][0], width)).map(e->e*Math.sqrt(Math.abs(weigths[0][0])) * Math.signum(weigths[0][0])).toArray();
		Ofilters[0][1] = Arrays.stream(gauss(spreads[0][1], width)).map(e->e*Math.sqrt(Math.abs(weigths[0][1])) * Math.signum(weigths[0][1])).toArray();
		Ofilters[0][2] = Arrays.stream(gauss(spreads[0][2], width)).map(e->e*Math.sqrt(Math.abs(weigths[0][2])) * Math.signum(weigths[0][2])).toArray();
		Ofilters[1] = new double[2][];
		Ofilters[1][0] = Arrays.stream(gauss(spreads[1][0], width)).map(e->e*Math.sqrt(Math.abs(weigths[1][0])) * Math.signum(weigths[1][0])).toArray();
		Ofilters[1][1] = Arrays.stream(gauss(spreads[1][1], width)).map(e->e*Math.sqrt(Math.abs(weigths[1][1])) * Math.signum(weigths[1][1])).toArray();
		Ofilters[2] = new double[2][];
		Ofilters[2][0] = Arrays.stream(gauss(spreads[2][0], width)).map(e->e*Math.sqrt(Math.abs(weigths[2][0])) * Math.signum(weigths[2][0])).toArray();
		Ofilters[2][1] = Arrays.stream(gauss(spreads[2][1], width)).map(e->e*Math.sqrt(Math.abs(weigths[2][1])) * Math.signum(weigths[2][1])).toArray();

		//Upsampling and downsampling
		if(uprate > 1)
		{
			//Generate upsampling kernel
			double[] upcol = new double[uprate*2-1];
			for(int i=0; i<upcol.length; i++)
			{
				upcol[i] = (uprate - Math.abs(uprate - i-1))/uprate;
			}
			//Resize it
			upcol = resize1D(upcol, upcol.length+width-1);
			//Convolve it with the opponent filters
			double[][][] ups = new double[3][][];
			ups[0]= new double[3][];
			ups[0][0] = conv1D(Ofilters[0][0], upcol);
			ups[0][1] = conv1D(Ofilters[0][1], upcol);
			ups[0][2] = conv1D(Ofilters[0][2], upcol);
			ups[1]= new double[2][];
			ups[1][0] = conv1D(Ofilters[1][0], upcol);
			ups[1][1] = conv1D(Ofilters[1][1], upcol);
			ups[2]= new double[2][];
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

	}

	/**
	 * Executes the sum of convolutions of different filters on the data in horizontal and vertical direction
	 * Padding is done by reflection
	 * @param data data to convolve
	 * @param filters filters
	 * @param w width of the data
	 * @return convoluted data
	 */
	public static double[] convolveSeparable(double[] data, double[][] filters, int w)
	{
		if(filters.length == 0)
			return Arrays.copyOf(data, data.length);
		double[] result = new double[data.length];
		double[] temp = new double[data.length];
		Array1DUtil.fill(result, 0.0);
		Array1DUtil.fill(temp, 0.0);
		int h = data.length/w;
		int xoff, yoff;
		for (double[] filter : filters) {
			int fmid = filter.length / 2;
			//Horizontal
			for (int x = 0; x < h; x++) {
				for (int y = 0; y < w; y++) {
					for (int foff = 0; foff < filter.length; foff++) {
						xoff = x;
						yoff = y - fmid + foff;
						if (yoff < 0) //Reflection
							yoff = -yoff - 1;
						if (yoff >= w)
							yoff = w - (yoff - w + 1);
						temp[x * w + y] += data[xoff * w + yoff] * filter[foff];
					}
				}
			}
			//Vertical
			for (int x = 0; x < h; x++) {
				for (int y = 0; y < w; y++) {
					for (int foff = 0; foff < filter.length; foff++) {
						xoff = x - fmid + foff;
						yoff = y;
						if (xoff < 0) //Reflection
							xoff = -xoff - 1;
						if (xoff >= h)
							xoff = h - (xoff - h + 1);
						result[x * w + y] += temp[xoff * w + yoff] * filter[foff];
					}
				}
			}
		}
		return result;
	}

	private static double[] conv1D(double[] data, double[] filter)
	{
		double[] result = new double[data.length];
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

	private static double[] resize1D(double[] src, int newSize)
	{
		double[] res = new double[newSize];
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

	private static double[] extractWithIndices(double[] src, int[] indices)
	{
		double[] result = new double[indices.length];
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
	public static double[] gauss(double halfwidth, int width)
	{
		double alpha = 2*Math.sqrt(Math.log(2))/(halfwidth-1);
		double[] result = new double[width];
		int offset = width/2;
		double sum=0;
		for(int i=0; i<width;i++)
		{
			result[i] = Math.exp(-alpha*alpha*(i-offset)*(i-offset));
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
	private static double[] matrixColorConvert(double[] C, double[][] M)
	{
		return new double[]{
				C[0]*M[0][0] + C[1]*M[0][1] + C[2]*M[0][2],
				C[1]*M[1][0] + C[1]*M[1][1] + C[2]*M[1][2],
				C[2]*M[2][0] + C[1]*M[2][1] + C[2]*M[2][2],
		};
	}

	private static double[] sRGBtoXYZ(double[] sRGB)
	{
		double[] RGB = {sRGB[0] <= 0.04045 ? sRGB[0]/12.92 : Math.pow((sRGB[0]+0.055)/1.055,2.4),
				sRGB[1] <= 0.04045 ? sRGB[1]/12.92 : Math.pow((sRGB[1]+0.055)/1.055,2.4),
				sRGB[2] <= 0.04045 ? sRGB[2]/12.92 : Math.pow((sRGB[2]+0.055)/1.055,2.4)};
		return matrixColorConvert(RGB, mSRGBtoXYZ);
	}

	private static double[] XYZtosRGB(double[] XYZ)
	{
		double[] RGB = matrixColorConvert(XYZ, mXYZtoSRGB);
		return new double[]{
				RGB[0] <= 0.0031308 ? RGB[0]*12.92 : Math.pow(RGB[0]*1.055, 1/2.4)-0.055,
				RGB[1] <= 0.0031308 ? RGB[1]*12.92 : Math.pow(RGB[1]*1.055, 1/2.4)-0.055,
				RGB[2] <= 0.0031308 ? RGB[2]*12.92 : Math.pow(RGB[2]*1.055, 1/2.4)-0.055,
		};
	}

	private static double[] XYZtoOpp(double[] XYZ)
	{
		return matrixColorConvert(XYZ, mXYZtoOpp);
	}

	private static double[] OppToXYZ(double[] Opp)
	{
		return matrixColorConvert(Opp, mOpptoXYZ);
	}

	private double[] XYZtoLAB(double[] XYZ)
	{
		double fx = LAB_f(XYZ[0]/ illuminant[0]);
		double fy = LAB_f(XYZ[1]/ illuminant[1]);
		double fz = LAB_f(XYZ[2]/ illuminant[2]);
		return new double[]{
				116*fy-16,
				500*(fx-fy),
				200*(fy-fz)
		};
	}

	private double[] LABtoXYZ(double[] LAB)
	{
		double L = (LAB[0]+16.0)/116.0;
		return new double[]
				{
						illuminant[0]*LAB_finv(L + LAB[1]/500.0),
						illuminant[1]*LAB_finv(L),
						illuminant[2]*LAB_finv(L - LAB[2]/200.0)
				};
	}

	private static double LAB_f(double t)
	{
		double delta = 6.0/29.0;
		return t > delta*delta*delta ? Math.pow(t, 1.0/3.0) : t/(3*delta*delta) + 4.0/29.0;
	}

	private static double LAB_finv(double t)
	{
		double delta = 6.0/29.0;
		return t > delta ? t*t*t : 3*delta*delta*(t-4.0/29.0);
	}

	/**
	 * Converts an sRBG image to its S-CIELAB representation
	 * @param image sRGB image in an [C][XY] array
	 * @param w width of the image in pixels
	 * @return Lab image in an [C][XY] array in its S-CIELAB representation
	 */
	public double[][] imageToScielab(double[][] image, int w)
	{
		double[][] result = new double[3][image[0].length];
		int h = image[0].length/(w);
		int offset;
		double[] srcBufferPixel = new double[3];
		double[] outBufferPixel;
		//First we convert to XYZ and then to Poirson&Wandell opponent
		for(int x=0; x<h; x++)
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
		}
		//Then we apply filters to mimic human vision
		result[0] = convolveSeparable(result[0], Ofilters[0], w);
		result[1] = convolveSeparable(result[1], Ofilters[1], w);
		result[2] = convolveSeparable(result[2], Ofilters[2], w);

		//Switch back to XYZ and then to LAB
		for(int x=0; x<h; x++)
		{
			for(int y=0; y<w; y++)
			{
				offset = x*w + y;
				srcBufferPixel[0] = result[0][offset];
				srcBufferPixel[1] = result[1][offset];
				srcBufferPixel[2] = result[2][offset];
				outBufferPixel = OppToXYZ(srcBufferPixel);
				result[0][offset] = outBufferPixel[0];
				result[1][offset] = outBufferPixel[1];
				result[2][offset] = outBufferPixel[2];
			}
		}
		return result;
	}

	public double[][] LabTosRGB(double[][] image, int w)
	{
		double[][] result = new double[3][image[0].length];
		int h = image[0].length/(w);
		int offset;
		double[] srcBufferPixel = new double[3];
		double[] outBufferPixel;
		for(int x=0; x<h; x++)
		{
			for(int y=0; y<w; y++)
			{
				offset = x*w + y;
				srcBufferPixel[0] = result[0][offset];
				srcBufferPixel[1] = result[1][offset];
				srcBufferPixel[2] = result[2][offset];
				outBufferPixel = XYZtosRGB(LABtoXYZ(srcBufferPixel));
				result[0][offset] = outBufferPixel[0];
				result[1][offset] = outBufferPixel[1];
				result[2][offset] = outBufferPixel[2];
			}
		}
		return result;
	}
}
