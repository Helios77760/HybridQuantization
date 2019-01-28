package plugins.dbrasseur.hybridquantization;

import icy.image.IcyBufferedImage;
import icy.image.colorspace.IcyColorSpace;
import icy.type.collection.array.Array1DUtil;
import icy.util.ColorUtil;

public class ScielabProcessor {
	private static final int minSAMPPERDEG = 224;
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
	private double[][][] Ofilters;
	private int uprate;

	public ScielabProcessor(Integer dpi, Double viewingDistance)
	{

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
