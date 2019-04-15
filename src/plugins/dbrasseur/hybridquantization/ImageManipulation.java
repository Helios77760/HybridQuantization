package plugins.dbrasseur.hybridquantization;

import com.nativelibs4java.opencl.*;
import com.ochafik.io.ReadText;
import icy.type.collection.array.Array1DUtil;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;

/**
 * Class for filtering with OpenCL
 * Derived from Alexandre Dufour's FilterToolbox plugin (icy.bioimageanalysis.org/plugin/Filter_Toolbox)
 */
public class ImageManipulation {
    public enum deltaETypes{CIE76, CIE94, CIEDE2000}

    private CLContext context;
    private CLQueue queue;
    private CLProgram program;

    private CLKernel convolution4Kernel;
    private CLKernel convolution1Kernel;
    private CLKernel RGB2XYZKernel;
    private CLKernel XYZ2RGBKernel;
    private CLKernel XYZ2OppKernel;
    private CLKernel Opp2LABKernel;
    private CLKernel QuantizeKernel;
    private CLKernel Quantize2OppKernel;
    private CLKernel DeltaEKernel;

    private float[][] filters4;
    private float[] absfilters4;
    private boolean openCLFiltersReady;

    private boolean openCLAvailable;

    public ImageManipulation(deltaETypes deltaEType){
        // Preparing the OpenCL system
        openCLAvailable=false;
        try{
            context = JavaCL.createBestContext();
            queue = context.createDefaultQueue();
            InputStream is = ImageManipulation.class.getResourceAsStream("OptimizedConvolution.cl");
            String programFile = ReadText.readText(is);
            program = context.createProgram(programFile);
            program.addBuildOption("-D"+deltaEType.name());
            program.build();
            convolution4Kernel = program.createKernel("convolve4Channels");
            convolution1Kernel = program.createKernel("convolve1Channel");
            RGB2XYZKernel = program.createKernel("RGB2XYZ");
            XYZ2RGBKernel = program.createKernel("XYZ2RGB");
            XYZ2OppKernel = program.createKernel("XYZ2Opp");
            Opp2LABKernel = program.createKernel("Opp2LAB");
            QuantizeKernel = program.createKernel("quantize");
            Quantize2OppKernel = program.createKernel("quantizeAndConvertToOpp");
            DeltaEKernel = program.createKernel("CIEDE");
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
            System.err.println(e.getMessage());
        } catch (UnsatisfiedLinkError linkError)
        {
            System.out.println("Warning (HybridQuantization): OpenCL drivers not found. Using basic Java implementation.");
        }
    }

    boolean getOpenCLAvailable()
    {
        return openCLAvailable;
    }

    public float[] RGBtoXYZ(float[] R,float[] G,float[] B)
    {
        float[] output = new float[4*R.length]; //float3 are aligned on 4*sizeof(float), see the OpenCL specification for details
        if(openCLAvailable)
        {
            CLEvent event;
            queue.flush();

            CLFloatBuffer cl_Rbuffer = context.createFloatBuffer(CLMem.Usage.Input, R.length);
            FloatBuffer R_buffer = cl_Rbuffer.map(queue, CLMem.MapFlags.Write).put(R);
            R_buffer.rewind();
            event = cl_Rbuffer.unmap(queue, R_buffer);

            CLFloatBuffer cl_Gbuffer = context.createFloatBuffer(CLMem.Usage.Input, G.length);
            FloatBuffer G_buffer = cl_Gbuffer.map(queue, CLMem.MapFlags.Write, event).put(G);
            G_buffer.rewind();
            event = cl_Gbuffer.unmap(queue, G_buffer);

            CLFloatBuffer cl_Bbuffer = context.createFloatBuffer(CLMem.Usage.Input, B.length);
            FloatBuffer B_buffer = cl_Bbuffer.map(queue, CLMem.MapFlags.Write, event).put(B);
            B_buffer.rewind();
            event = cl_Bbuffer.unmap(queue, B_buffer);
            
            FloatBuffer outBuffer = ByteBuffer.allocateDirect(4*4*R.length).order(context.getByteOrder()).asFloatBuffer();
            CLFloatBuffer cl_outBuffer = context.createFloatBuffer(CLMem.Usage.Output, outBuffer, false);

            RGB2XYZKernel.setArgs(cl_Rbuffer, cl_Gbuffer, cl_Bbuffer, cl_outBuffer);

            event = RGB2XYZKernel.enqueueNDRange(queue, new int[]{R.length}, event);
            event = cl_outBuffer.read(queue, outBuffer, true, event);

            queue.finish();
            outBuffer.get(output);

            cl_Rbuffer.release();
            cl_Gbuffer.release();
            cl_Bbuffer.release();
            cl_outBuffer.release();
            return output;
        }

        //Java mode
        for(int i=0; i<R.length; i++)
        {
            float[] XYZ =ScielabProcessor.sRGBtoXYZ(new float[]{R[i], G[i], B[i]});
            int off = i<<2;
            output[off] = XYZ[0];
            output[off+1] = XYZ[1];
            output[off+2] = XYZ[2];
            output[off+3] = 0.0f;
        }

        return output;
    }

    public float[][] XYZtoRGB(float[] XYZ)
    {
        float[][] RGB = new float[3][XYZ.length/4];
        if(openCLAvailable)
        {
            CLEvent event;
            queue.flush();

            FloatBuffer R_buffer = ByteBuffer.allocateDirect(XYZ.length).order(context.getByteOrder()).asFloatBuffer();
            CLFloatBuffer cl_Rbuffer = context.createFloatBuffer(CLMem.Usage.Output, R_buffer, false);

            FloatBuffer G_buffer = ByteBuffer.allocateDirect(XYZ.length).order(context.getByteOrder()).asFloatBuffer();
            CLFloatBuffer cl_Gbuffer = context.createFloatBuffer(CLMem.Usage.Output, G_buffer, false);

            FloatBuffer B_buffer = ByteBuffer.allocateDirect(XYZ.length).order(context.getByteOrder()).asFloatBuffer();
            CLFloatBuffer cl_Bbuffer = context.createFloatBuffer(CLMem.Usage.Output, B_buffer, false);

            CLFloatBuffer cl_inBuffer = context.createFloatBuffer(CLMem.Usage.Input,XYZ.length);
            FloatBuffer inBuffer = cl_inBuffer.map(queue, CLMem.MapFlags.Write).put(XYZ);
            inBuffer.rewind();
            event = cl_inBuffer.unmap(queue, inBuffer);

            /*FloatBuffer outBuffer = ByteBuffer.allocateDirect(4*XYZ.length).order(context.getByteOrder()).asFloatBuffer();
            CLFloatBuffer cl_outBuffer = context.createFloatBuffer(CLMem.Usage.Output, outBuffer, false);*/


            XYZ2RGBKernel.setArgs(cl_inBuffer, cl_Rbuffer, cl_Gbuffer, cl_Bbuffer);
            event = XYZ2RGBKernel.enqueueNDRange(queue, new int[]{XYZ.length/4}, event);

            cl_Rbuffer.read(queue, R_buffer, false, event);
            cl_Gbuffer.read(queue, G_buffer, false, event);
            cl_Bbuffer.read(queue, B_buffer, true, event);

            queue.finish();
            R_buffer.get(RGB[0]);
            G_buffer.get(RGB[1]);
            B_buffer.get(RGB[2]);

            cl_Rbuffer.release();
            cl_Gbuffer.release();
            cl_Bbuffer.release();
            cl_inBuffer.release();

            return RGB;
        }

        //Java mode
        for(int i=0; i<XYZ.length/4; i++)
        {
            int off=i<<2;
            float[] RGBi =ScielabProcessor.XYZtosRGB(new float[]{XYZ[off], XYZ[off+1], XYZ[off+2]});
            RGB[0][i] = RGBi[0];
            RGB[1][i] = RGBi[1];
            RGB[2][i] = RGBi[2];
        }

        return RGB;
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
    public static float[] convolveSeparable(float[] data, float[][] filters, int w)
    {
        if(filters.length == 0)
            return Arrays.copyOf(data, data.length);
        float[] result = new float[data.length];
        float[] temp = new float[data.length];
        Array1DUtil.fill(result, 0.0);
        Array1DUtil.fill(temp, 0.0);
        int h = data.length/w;
        int xoff, yoff;
        for (float[] filter : filters) {
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
            float[] absFilter = Array1DUtil.doubleArrayToFloatArray(Arrays.stream(Array1DUtil.floatArrayToDoubleArray(filter)).map(Math::abs).toArray());
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

    private CLEvent loadGPUBuffer(CLFloatBuffer cl_buffer, float[] data, CLEvent... eventsToWait)
    {
        FloatBuffer buffer;
        if(eventsToWait.length > 0)
        {
            buffer = cl_buffer.map(queue, CLMem.MapFlags.Write, eventsToWait).put(data);
        }else
        {
            buffer = cl_buffer.map(queue, CLMem.MapFlags.Write).put(data);
        }
        buffer.rewind();
        return cl_buffer.unmap(queue, buffer);
    }

    public float[] XYZtoScielab(float[] XYZ, float[][][] filters, float[] absfilters,int w, float[] illuminant) {

        float[] lab = new float[XYZ.length];
        int h=(XYZ.length/4)/w;
        if(openCLAvailable)
        {
            CLEvent event;
            queue.flush();

            if(!openCLFiltersReady)
            {
                updateOpenCLFilters(filters, absfilters);
            }
            int[] workSize = {XYZ.length / 4};
            int filterLength = filters4[0].length;
            int filterHalfWidth = filters4[0].length/8;

            //Inputs on GPU
            CLFloatBuffer cl_inBuffer = context.createFloatBuffer(CLMem.Usage.Input,XYZ.length);
            event = loadGPUBuffer(cl_inBuffer, XYZ);

            //Buffers on GPU
            //FloatBuffer oppBuffer = ByteBuffer.allocateDirect(XYZ.length).order(context.getByteOrder()).asFloatBuffer();
            CLFloatBuffer cl_OppBuffer = context.createFloatBuffer(CLMem.Usage.InputOutput, XYZ.length);

            //FloatBuffer convBuffer = ByteBuffer.allocateDirect(XYZ.length).order(context.getByteOrder()).asFloatBuffer();
            CLFloatBuffer cl_convBuffer = context.createFloatBuffer(CLMem.Usage.InputOutput, XYZ.length);

            //FloatBuffer tempBuffer = ByteBuffer.allocateDirect(XYZ.length).order(context.getByteOrder()).asFloatBuffer();
            CLFloatBuffer cl_tempBuffer = context.createFloatBuffer(CLMem.Usage.InputOutput, XYZ.length);

            CLFloatBuffer cl_filterBuffer = context.createFloatBuffer(CLMem.Usage.Input,filterLength);

            //Conversion entre XYZ et Opp
            XYZ2OppKernel.setArgs(cl_inBuffer, cl_OppBuffer);
            event = XYZ2OppKernel.enqueueNDRange(queue, workSize,event);

            //Premier filtre
            event = loadGPUBuffer(cl_filterBuffer, filters4[0], event);

            convolution4Kernel.setArgs(cl_OppBuffer, cl_filterBuffer, filterHalfWidth, w, h, 0, cl_tempBuffer);
            event = convolution4Kernel.enqueueNDRange(queue, workSize, event);

            convolution4Kernel.setArgs(cl_tempBuffer, cl_filterBuffer, filterHalfWidth, h, w, 0, cl_convBuffer);
            event = convolution4Kernel.enqueueNDRange(queue, workSize, event);

            //Deuxieme filtre
            event = loadGPUBuffer(cl_filterBuffer, filters4[1], event);

            convolution4Kernel.setArgs(cl_OppBuffer, cl_filterBuffer, filterHalfWidth, w, h, 0, cl_tempBuffer);
            event = convolution4Kernel.enqueueNDRange(queue, workSize, event);

            convolution4Kernel.setArgs(cl_tempBuffer, cl_filterBuffer, filterHalfWidth, h, w, 1, cl_convBuffer);
            event = convolution4Kernel.enqueueNDRange(queue, workSize, event);

            //Troisieme filtre
            event = loadGPUBuffer(cl_filterBuffer, filters4[2], event);

            convolution1Kernel.setArgs(cl_OppBuffer, cl_filterBuffer, filterHalfWidth, w, h, 0, cl_tempBuffer);
            event = convolution1Kernel.enqueueNDRange(queue, workSize, event);

            event = loadGPUBuffer(cl_filterBuffer, absfilters4, event);

            convolution1Kernel.setArgs(cl_tempBuffer, cl_filterBuffer, filterHalfWidth, h, w, 1, cl_convBuffer);
            event = convolution1Kernel.enqueueNDRange(queue, workSize, event);

            //Buffers on host
            FloatBuffer labBuffer = ByteBuffer.allocateDirect(XYZ.length*4).order(context.getByteOrder()).asFloatBuffer();
            CLFloatBuffer cl_labBuffer = context.createFloatBuffer(CLMem.Usage.Output, labBuffer, false);

            Opp2LABKernel.setArgs(cl_convBuffer, illuminant[0], illuminant[1], illuminant[2], cl_labBuffer);
            event = Opp2LABKernel.enqueueNDRange(queue, workSize, event);
            queue.finish();
            cl_labBuffer.read(queue,labBuffer, true, event);


            labBuffer.get(lab);

            cl_convBuffer.release();
            cl_filterBuffer.release();
            cl_inBuffer.release();
            cl_labBuffer.release();
            cl_OppBuffer.release();
            cl_tempBuffer.release();

            return lab;
        }
        return lab;
    }

    /**
     * Returns the optimal color palette to quantify the image
     * @param inlinergbOriginal original RGB image ine the format RGBARGBA...
     * @param inlineScielabOriginal originalimage obtained by scielab transformation in the format RGBARGBA...
     * @param w width of the image
     * @param nbOfColors number of colors
     * @param simulatedAnnealing simulated annealing parameters and functions
     * @param filters filters for the scielab convolution
     * @param absfilters absolute of the last filter of channel Opp1
     * @return optimal colors used in the quantization, in RGBARGBA.... format
     */
    public float[] findBestQuantization(float[] inlinergbOriginal, float[] inlineScielabOriginal, int w,int nbOfColors, SWASA simulatedAnnealing, float[][][] filters, float[] absfilters, float[] illuminant)
    {
        simulatedAnnealing.reset();
        float[] colors = simulatedAnnealing.generateRandomColors(nbOfColors);
        float[] currentColors = new float[colors.length];
        float[] bestColors = Arrays.copyOf(colors, colors.length);
        double bestError;
        int h = (inlinergbOriginal.length/4)/w;
        float[] errorArray = new float[inlinergbOriginal.length/4];
        int[] usedColors = new int[nbOfColors];
        if(openCLAvailable)
        {
            CLEvent loadcomp, loadrgb, loadfilter1, loadfilter2, loadfilter3, loadabs, loadcolors;
            queue.flush();

            if(!openCLFiltersReady)
            {
                updateOpenCLFilters(filters, absfilters);
            }
            int[] workSize = {inlinergbOriginal.length / 4};
            int filterLength = filters4[0].length;
            int filterHalfWidth = filters4[0].length/8;
            double currentError, error;

            //BUFFER DECLARATION

            //Input buffers
            CLFloatBuffer cl_comparisonBuffer = context.createFloatBuffer(CLMem.Usage.Input,inlineScielabOriginal.length);
            CLFloatBuffer cl_rgbBuffer = context.createFloatBuffer(CLMem.Usage.Input,inlinergbOriginal.length);
            CLFloatBuffer cl_colorBuffer = context.createFloatBuffer(CLMem.Usage.Input, colors.length);

            //Filter buffers
            CLFloatBuffer cl_filterBuffer = context.createFloatBuffer(CLMem.Usage.Input,filterLength);
            CLFloatBuffer cl_filterBuffer2 = context.createFloatBuffer(CLMem.Usage.Input,filterLength);
            CLFloatBuffer cl_filterBuffer3 = context.createFloatBuffer(CLMem.Usage.Input,filterLength);
            CLFloatBuffer cl_absfilterBuffer = context.createFloatBuffer(CLMem.Usage.Input,filterLength);

            //Middle buffers
            CLFloatBuffer cl_OppBuffer = context.createFloatBuffer(CLMem.Usage.InputOutput, inlineScielabOriginal.length);
            CLFloatBuffer cl_tempBuffer = context.createFloatBuffer(CLMem.Usage.InputOutput, inlineScielabOriginal.length);
            CLFloatBuffer cl_convBuffer = context.createFloatBuffer(CLMem.Usage.InputOutput, inlineScielabOriginal.length);

            //Utility buffer
            int[] initialUsedColors = new int[nbOfColors];
            IntBuffer usedColorBuffer = ByteBuffer.allocateDirect(nbOfColors*4).order(context.getByteOrder()).asIntBuffer();
            CLIntBuffer cl_usedColorBuffer = context.createIntBuffer(CLMem.Usage.Output, usedColorBuffer, false);
            FloatBuffer errorBuffer = ByteBuffer.allocateDirect(inlineScielabOriginal.length*4).order(context.getByteOrder()).asFloatBuffer();
            CLFloatBuffer cl_errorBuffer = context.createFloatBuffer(CLMem.Usage.Output, errorBuffer, false);

            //BUFFER LOADING

            //Input buffers
            loadcomp = loadGPUBuffer(cl_comparisonBuffer,inlineScielabOriginal);
            loadrgb = loadGPUBuffer(cl_rgbBuffer, inlinergbOriginal);
            loadcolors = loadGPUBuffer(cl_colorBuffer, colors);

            //Filter buffers
            loadfilter1 = loadGPUBuffer(cl_filterBuffer, filters4[0]);
            loadfilter2 = loadGPUBuffer(cl_filterBuffer2, filters4[1]);
            loadfilter3 = loadGPUBuffer(cl_filterBuffer3, filters4[2]);
            loadabs = loadGPUBuffer(cl_absfilterBuffer, absfilters4);

            //ARGUMENTS DEFINITION FOR CONSTANT KERNELS
            Quantize2OppKernel.setArgs(cl_rgbBuffer, cl_colorBuffer, nbOfColors, cl_usedColorBuffer, cl_OppBuffer);
            DeltaEKernel.setArgs(cl_comparisonBuffer, cl_tempBuffer, cl_errorBuffer);
            Opp2LABKernel.setArgs(cl_OppBuffer, illuminant[0], illuminant[1], illuminant[2], cl_tempBuffer);

            CLEvent.waitFor(loadcomp, loadrgb, loadcolors, loadfilter1, loadfilter2, loadfilter3, loadabs);
            bestError = currentError = computeQuantizationError(cl_comparisonBuffer, cl_rgbBuffer, cl_colorBuffer,cl_filterBuffer, cl_filterBuffer2, cl_filterBuffer3, cl_absfilterBuffer, cl_OppBuffer, cl_tempBuffer, cl_convBuffer, cl_usedColorBuffer, usedColorBuffer,usedColors, cl_errorBuffer, errorBuffer, errorArray, w, h, workSize, filterHalfWidth, simulatedAnnealing);

            //MAIN LOOP
            int maxiter = simulatedAnnealing.getImax();
            for(int i=1; i<=maxiter;i++)
            {
                usedColorBuffer.rewind();
                usedColorBuffer.put(initialUsedColors);
                usedColorBuffer.rewind();
                CLEvent event = cl_usedColorBuffer.write(queue,usedColorBuffer, false);
                simulatedAnnealing.reduceTemperatureIfNecessary(i);
                simulatedAnnealing.generateNeighboringColors(colors, currentColors,nbOfColors, i);
                loadGPUBuffer(cl_colorBuffer, currentColors, event).waitFor();
                error = computeQuantizationError(cl_comparisonBuffer, cl_rgbBuffer, cl_colorBuffer,cl_filterBuffer, cl_filterBuffer2, cl_filterBuffer3, cl_absfilterBuffer, cl_OppBuffer, cl_tempBuffer, cl_convBuffer, cl_usedColorBuffer, usedColorBuffer,usedColors, cl_errorBuffer, errorBuffer, errorArray, w, h, workSize, filterHalfWidth, simulatedAnnealing);
                if(simulatedAnnealing.isAccepted(error-currentError))
                {
                    currentError = error;
                    System.arraycopy(currentColors, 0, colors,0, currentColors.length);
                    if(currentError < bestError)
                    {
                        System.arraycopy(currentColors, 0, bestColors, 0, currentColors.length);
                        bestError = currentError;
                    }
                }
                if(i % 50 == 0)
                {
                    System.out.println(i + " error : " + bestError);
                }
            }

            //Releasing the buffers
            cl_absfilterBuffer.release();
            cl_colorBuffer.release();
            cl_comparisonBuffer.release();
            cl_convBuffer.release();
            cl_errorBuffer.release();
            cl_filterBuffer.release();
            cl_filterBuffer2.release();
            cl_filterBuffer3.release();
            cl_OppBuffer.release();
            cl_rgbBuffer.release();
            cl_tempBuffer.release();
            cl_usedColorBuffer.release();
        }
        return bestColors;
    }

    private double computeQuantizationError(CLFloatBuffer original, CLFloatBuffer rgb, CLFloatBuffer colors, CLFloatBuffer filter1, CLFloatBuffer filter2, CLFloatBuffer filter3, CLFloatBuffer absfilter, CLFloatBuffer opp, CLFloatBuffer temp, CLFloatBuffer conv, CLIntBuffer usedColorBuffer, IntBuffer usedColors,int[] usedColorsArray, CLFloatBuffer errorBuffer, FloatBuffer errors, float[] errorArray, int w, int h, int[] worksize, int filterHalfWidth, SWASA swasa) {
        CLEvent event;
        //Quantification
        event = Quantize2OppKernel.enqueueNDRange(queue, worksize);
        usedColorBuffer.read(queue, usedColors, false, event);

        //Premier filtre
        convolution4Kernel.setArgs(opp, filter1, filterHalfWidth, w, h, 0, temp);
        event = convolution4Kernel.enqueueNDRange(queue, worksize, event);
        convolution4Kernel.setArgs(temp, filter1, filterHalfWidth, h, w, 0, conv);
        event = convolution4Kernel.enqueueNDRange(queue, worksize, event);

        //Deuxième filtre
        convolution4Kernel.setArgs(opp, filter2, filterHalfWidth, w, h, 0, temp);
        event = convolution4Kernel.enqueueNDRange(queue, worksize, event);
        convolution4Kernel.setArgs(temp, filter2, filterHalfWidth, h, w, 1, conv);
        event = convolution4Kernel.enqueueNDRange(queue, worksize, event);

        //Troisieme filtre sur le premier canal
        convolution1Kernel.setArgs(opp, filter3, filterHalfWidth, w, h, 0, temp);
        event = convolution1Kernel.enqueueNDRange(queue, worksize, event);
        convolution1Kernel.setArgs(temp, absfilter, filterHalfWidth, h, w, 1, conv);
        event = convolution1Kernel.enqueueNDRange(queue, worksize, event);

        //Conversion en lab
        event = Opp2LABKernel.enqueueNDRange(queue, worksize, event);

        //Calcul de l'erreur
        event = DeltaEKernel.enqueueNDRange(queue, worksize, event);
        queue.finish();
        event = errorBuffer.read(queue,errors, true, event);
        errors.rewind();
        errors.get(errorArray);
        usedColors.rewind();
        usedColors.get(usedColorsArray);

        /*for(int i=0; i<usedColorsArray.length;i++)
        {
            System.out.print(usedColorsArray[i]+" ");
        }
        /*for(int i=0; i<errorArray.length;i++)
        {
            if(Double.isNaN(errorArray[i]))
                System.out.print(errorArray[i]+" ");
        }*/
        double error = averageArray(errorArray)/3;
        //System.out.println(error);
        return error + swasa.computePenalty(usedColorsArray);
    }

    /**
     * Returns the average of the array
     * @param array array
     * @return average
     */
    private double averageArray(float[] array)
    {
        return sumArray(array, 0, array.length, 32 - Integer.numberOfLeadingZeros(Runtime.getRuntime().availableProcessors()))/array.length; //Profondeur = ceil(log2(nombre de processeurs))
    }

    private double sumArray(float[] array, int start, int end, int recursionProf)
    {
        if(end <= start)
            return 0;
        if(recursionProf <= 0) //Recursion max, on calcule de manière séquentielle
        {
            double sum=0;
            for(int i=start; i<end;i++)
            {
                sum+=array[i];
            }
            return sum;
        }else
        {
            double[] results = new double[2];
            Thread a = new Thread(()->results[0] = sumArray(array,start, (start+end)/2, recursionProf-1));
            Thread b = new Thread(()->results[1] = sumArray(array,(start+end)/2, end, recursionProf-1));

            a.start();
            b.start();

            try{
                a.join();
                b.join();
            }catch(InterruptedException ignored) {}
            return results[0]+results[1];
        }
    }

    public float[] quantize(float[] inlineImageRGB, float[] colors)
    {
        float[] quantizedImage = new float[inlineImageRGB.length];
        if(openCLAvailable)
        {
            System.out.println("q");
            CLEvent event;
            CLFloatBuffer cl_rgbBuffer = context.createFloatBuffer(CLMem.Usage.Input,inlineImageRGB.length);
            event = loadGPUBuffer(cl_rgbBuffer, inlineImageRGB);
            CLFloatBuffer cl_colorsBuffer = context.createFloatBuffer(CLMem.Usage.Input,colors.length);
            event = loadGPUBuffer(cl_colorsBuffer, colors, event);
            CLIntBuffer cl_usedColorsBuffer = context.createIntBuffer(CLMem.Usage.Output,colors.length/4);

            FloatBuffer outBuffer = ByteBuffer.allocateDirect(inlineImageRGB.length*4).order(context.getByteOrder()).asFloatBuffer();
            CLFloatBuffer cl_outBuffer = context.createFloatBuffer(CLMem.Usage.Output, outBuffer, false);

            QuantizeKernel.setArgs(cl_rgbBuffer, cl_colorsBuffer, colors.length/4, cl_usedColorsBuffer, cl_outBuffer);
            event = QuantizeKernel.enqueueNDRange(queue,new int[]{inlineImageRGB.length/4}, event);
            queue.finish();
            cl_outBuffer.read(queue,outBuffer, true, event);
            outBuffer.get(quantizedImage);

            cl_outBuffer.release();
            cl_colorsBuffer.release();
            cl_rgbBuffer.release();
            cl_usedColorsBuffer.release();
        }

        return quantizedImage;
    }

    public void updateOpenCLFilters(float[][][] filters, float[] absfilters) {
        filters4 = new float[3][];
        absfilters4 = new float[absfilters.length*4];
        int off;
        for(int i=0; i<2; i++)
        {
            filters4[i] = new float[filters[0][i].length*4];
            for(int j=0; j < filters[0][i].length; j++)
            {
                off = j<<2;
                filters4[i][off] = filters[0][i][j];
                filters4[i][off+1] = filters[1][i][j];
                filters4[i][off+2] = filters[2][i][j];
                filters4[i][off+3] = 0.0f;
            }
        }
        filters4[2] = new float[filters[0][2].length*4];
        for(int j=0; j < filters[0][2].length; j++)
        {
            off = j<<2;
            filters4[2][off] = filters[0][2][j];
            filters4[2][off+1] = 0.0f;
            filters4[2][off+2] = 0.0f;
            filters4[2][off+3] = 0.0f;
        }

        for(int j=0; j < absfilters.length; j++)
        {
            off = j<<2;
            absfilters4[off] = absfilters[j];
            absfilters4[off+1] = 0.0f;
            absfilters4[off+2] = 0.0f;
            absfilters4[off+3] = 0.0f;
        }
        openCLFiltersReady=true;
    }
}
