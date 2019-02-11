package plugins.dbrasseur.hybridquantization;

import com.nativelibs4java.opencl.*;
import com.ochafik.io.ReadText;
import icy.image.IcyBufferedImage;
import icy.image.IcyBufferedImageUtil;
import icy.main.Icy;
import icy.sequence.Sequence;
import icy.type.DataType;
import icy.type.collection.array.Array1DUtil;
import plugins.adufour.filtering.Convolution1D;
import plugins.adufour.filtering.ConvolutionCL;
import plugins.adufour.filtering.ConvolutionException;
import plugins.adufour.filtering.Kernels1D;
import plugins.adufour.vars.lang.VarBoolean;

import java.io.IOException;
import java.util.Arrays;

/**
 * Class for filtering with OpenCL
 * Derived from Alexandre Dufour's FilterToolbox plugin (icy.bioimageanalysis.org/plugin/Filter_Toolbox)
 */
public class ImageManipulation {

    private ConvolutionCL convolutionCL;
    private CLContext context;
    private CLQueue queue;
    private CLProgram program;

    private boolean openCLAvailable;

    public ImageManipulation(){
        // Preparing the OpenCL system
        openCLAvailable=false;
        try{
            context = JavaCL.createBestContext();
            queue = context.createDefaultQueue();
            String programFile = ReadText.readText(ConvolutionCL.class.getResourceAsStream("Convolution.cl"));
            program = context.createProgram(programFile).build();
            convolutionCL = new ConvolutionCL(context, program, queue);
            openCLAvailable=true;
        }catch (IOException e)
        {
            System.out.println("Warning (HybridQuantization): unable to load the OpenCL code. Continuing in pure Java mode.");
            e.printStackTrace();
        }
        catch (CLException | CLBuildException | NoClassDefFoundError e)
        {
            System.out.println("Warning (HybridQuantization): unable to create the OpenCL context. Continuing in pure Java mode.");
            e.printStackTrace();
        } catch (UnsatisfiedLinkError linkError)
        {
            System.out.println("Warning (HybridQuantization): OpenCL drivers not found. Using basic Java implementation.");
        }
    }

    public void convolve(double[][] input, double[][][] filters, int w)
    {
        boolean openCLFailed=false;
        int h = input[0].length/w;
        if(openCLAvailable)
        {
            Sequence bufferSeq = new Sequence();
            Sequence kernel;
            IcyBufferedImage im = new IcyBufferedImage(w,h,1, DataType.DOUBLE);
            Kernels1D k1d = Kernels1D.CUSTOM;
            VarBoolean stopFlag = new VarBoolean("stopflag", false); //TODO Replace stopflag by actual button

            try{
                double[][] temp = new double[3][input[0].length];
                bufferSeq.addImage(im);
                for(int c=0; c<input.length;c++)
                {
                    Arrays.fill(temp[c], 0);
                    im.setDataXY(0, Array1DUtil.doubleArrayToArray(input[c], im.getDataXY(0)));
                    if(filters.length>=c)
                    {
                        for(int f=0;  f < filters[c].length;f++)
                        {
                            kernel = k1d.createCustomKernel1D(filters[c][f], true).toSequence();
                            convolutionCL.convolve(bufferSeq, kernel, false, 1, stopFlag);
                            Sequence kernelY = new Sequence();
                            kernelY.addImage(new IcyBufferedImage(1, kernel.getSizeX(), 1, kernel.getDataType_()));
                            System.arraycopy(Arrays.stream(kernel.getDataXYAsDouble(0, 0, 0)).map(Math::abs).toArray(), 0, kernelY.getDataXY(0, 0, 0), 0, kernel.getSizeX());
                            convolutionCL.convolve(bufferSeq, kernelY, false, 1, stopFlag);
                            double[] finalTempBuffer = bufferSeq.getDataXYAsDouble(0,0,0);
                            int finalC = c;
                            Arrays.parallelSetAll(temp[finalC], i->temp[finalC][i]+ finalTempBuffer[i]);
                        }
                    }
                }
                input[0] = Arrays.copyOf(temp[0],temp[0].length);
                input[1] = Arrays.copyOf(temp[1],temp[1].length);
                input[2] = Arrays.copyOf(temp[2],temp[2].length);
            }catch (Exception e)
            {
                openCLFailed=true;
                e.printStackTrace();
                System.err.println("WARNING: Unable to run in OpenCL mode. Continuing in CPU mode.");
            }
        }

        if(!openCLAvailable || openCLFailed)
        {
            /*double [][] zxy = new double[1][];
            zxy[0] = Arrays.copyOf(input[0], input[0].length);
            try {
                Convolution1D.convolve(zxy, w, h, filters[0][0], Arrays.stream(filters[0][0]).map(Math::abs).toArray(), null);
            } catch (ConvolutionException e) {
                e.printStackTrace();
            }*/
            input[0] = convolveSeparable(input[0], filters[0], w);
            input[1] = convolveSeparable(input[1], filters[1], w);
            input[2] = convolveSeparable(input[2], filters[2], w);
        }
    }

    /**
     * Executes the sum of convolutions of different filters on the data in horizontal and vertical direction
     * Padding is done by reflection
     * The vertical kernel is the absolute of the horizontal one
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
                        else if (yoff >= w)
                            yoff = w - (yoff - w + 1);
                        temp[x * w + y] += data[xoff * w + yoff] * filter[foff];
                    }
                }
            }
            double[] absFilter = Arrays.stream(filter).map(Math::abs).toArray();
            //Vertical
            for (int x = 0; x < h; x++) {
                for (int y = 0; y < w; y++) {
                    for (int foff = 0; foff < filter.length; foff++) {
                        xoff = x - fmid + foff;
                        yoff = y;
                        if (xoff < 0) //Reflection
                            xoff = -xoff - 1;
                        else if (xoff >= h)
                            xoff = h - (xoff - h + 1);
                        result[x * w + y] += temp[xoff * w + yoff] * absFilter[foff];
                    }
                }
            }
        }
        return result;
    }

    public void close()
    {
        if(queue != null) queue.release();
        if (context != null) context.release();
    }
}
