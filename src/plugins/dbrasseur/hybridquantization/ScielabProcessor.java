package plugins.dbrasseur.hybridquantization;

import icy.image.IcyBufferedImage;
import icy.image.colorspace.IcyColorSpace;
import icy.type.collection.array.Array1DUtil;
import icy.util.ColorUtil;

import java.util.Arrays;

/**
 * Class for S-cielab processing
 * S-CIELAB values and code are derived from the S-CIELAB example code http://scarlet.stanford.edu/~brian/scielab/scielab.html
 * XYZ-Opp conversion matrices are from A HYBRID COLOR QUANTIZATION ALGORITHM INCORPORATING A HUMAN VISUAL PERCEPTION MODEL
 */
public class ScielabProcessor {
	private static final int minSAMPPERDEG = 224;   //As specified in the S-CIELAB example code :
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

	public ScielabProcessor(int dpi, double viewingDistance)
	{
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
		int finalSampPerDeg = sampPerDeg;
		double[][] spreads = (double[][])Arrays.stream(halfwidths).map(arr-> Arrays.stream(arr).map(e-> e* finalSampPerDeg).toArray()).toArray();

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


	}

	public static double[] convolve(double[] data, double[] filter, int dataw, int filterw)
	{
		//TODO
		return null;
	}

	private static double[] conv1D(double[] data, double[] filter)
	{
		//TODO
		return null;
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
	private double[] matrixColorConvert(double[] C, double[][] M)
	{
		return new double[]{
				C[0]*M[0][0] + C[1]*M[0][1] + C[2]*M[0][2],
				C[1]*M[1][0] + C[1]*M[1][1] + C[2]*M[1][2],
				C[2]*M[2][0] + C[1]*M[2][1] + C[2]*M[2][2],
		};
	}

	private double[] sRGBtoXYZ(double[] sRGB)
	{
		double[] RGB = {sRGB[0] <= 0.04045 ? sRGB[0]/12.92 : Math.pow((sRGB[0]+0.055)/1.055,2.4),
				sRGB[1] <= 0.04045 ? sRGB[1]/12.92 : Math.pow((sRGB[1]+0.055)/1.055,2.4),
				sRGB[2] <= 0.04045 ? sRGB[2]/12.92 : Math.pow((sRGB[2]+0.055)/1.055,2.4)};
		return matrixColorConvert(RGB, mSRGBtoXYZ);
	}

	private double[] XYZtosRGB(double[] XYZ)
	{
		double[] RGB = matrixColorConvert(XYZ, mXYZtoSRGB);
		return new double[]{
				RGB[0] <= 0.0031308 ? RGB[0]*12.92 : Math.pow(RGB[0]*1.055, 1/2.4)-0.055,
				RGB[1] <= 0.0031308 ? RGB[1]*12.92 : Math.pow(RGB[1]*1.055, 1/2.4)-0.055,
				RGB[2] <= 0.0031308 ? RGB[2]*12.92 : Math.pow(RGB[2]*1.055, 1/2.4)-0.055,
		};
	}

	private double[] XYZtoOpp(double[] XYZ)
	{
		return matrixColorConvert(XYZ, mXYZtoOpp);
	}

	private double[] OppToXYZ(double[] Opp)
	{
		return matrixColorConvert(Opp, mOpptoXYZ);
	}

	/**
	 * Converts an sRBG image to its S-CIELAB representation
	 * @param image sRGB image in an [XYC] array
	 * @param w width of the image in pixels
	 * @return Lab image in an [XYC] array in its S-CIELAB representation
	 */
	public double[] imageToScielab(double[] image, int w)
	{
		IcyColorSpace cs = new IcyColorSpace(3);
		double[] result = new double[image.length];
		int h = image.length/(3*w);
		int offset;
		double[] srcBufferPixel = new double[3];
		double[] outBufferPixel;
		//First we convert to XYZ and then to Poirson&Wandell opponent
		for(int x=0; x<h; x++)
		{
			for(int y=0; y<w; y++)
			{
				offset = (x*w + y)*3;
				srcBufferPixel[0] = (float)image[offset];
				srcBufferPixel[1] = (float)image[offset+1];
				srcBufferPixel[2] = (float)image[offset+2];
				outBufferPixel = XYZtoOpp(sRGBtoXYZ(srcBufferPixel));
				result[offset] = outBufferPixel[0];
				result[offset+1] = outBufferPixel[1];
				result[offset+2] = outBufferPixel[2];
			}
		}
		//Then we apply filters to mimic human vision

		return result;
	}
}
