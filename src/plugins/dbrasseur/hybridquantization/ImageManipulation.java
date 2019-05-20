package plugins.dbrasseur.hybridquantization;

import com.nativelibs4java.opencl.*;
import com.ochafik.io.ReadText;
import icy.type.collection.array.Array1DUtil;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
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

    private CLKernel convolutionScielabTemp;
    private CLKernel convolutionScielabEnd;

    private float[][] filters4;
    private float[] absfilters4;

    private float[] filter3;
    private float[] absfilter3;

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

            convolutionScielabTemp = program.createKernel("computeScielabKernelsTemp");
            convolutionScielabEnd = program.createKernel("computeScielabKernelsEnd");

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
            CLFloatBuffer cl_OppBuffer = context.createFloatBuffer(CLMem.Usage.InputOutput, XYZ.length);

            CLFloatBuffer cl_convBuffer = context.createFloatBuffer(CLMem.Usage.InputOutput, XYZ.length);

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
        //float[] colors = simulatedAnnealing.generateRandomColors(nbOfColors);
        //float[] currentColors = new float[colors.length];

        float[][] colors;
        float[][] currentColors;

        float[] bestColors = new float[nbOfColors*4];
        double bestError;
        int h = (inlinergbOriginal.length/4)/w;
        //float[] errorArray = new float[inlinergbOriginal.length/4];
        //int[] usedColors = new int[nbOfColors];
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
            int populationSize = simulatedAnnealing.getPopulationSize();
            CLEvent[][] events = new CLEvent[populationSize][9];

            colors = new float[populationSize][];
            for(int i=0; i<populationSize; i++)
            {
                colors[i] = simulatedAnnealing.generateRandomColors(nbOfColors);
            }
            currentColors = new float[populationSize][colors[0].length];
            bestColors = new float[colors[0].length];



            //BUFFER DECLARATION

            //Utility buffer
            int[] initialUsedColors = new int[nbOfColors];

            IntBuffer[] usedColorBuffers = new IntBuffer[populationSize];
            CLIntBuffer[] cl_usedColorBuffers = new CLIntBuffer[populationSize];
            FloatBuffer[] errorBuffers = new FloatBuffer[populationSize];
            CLFloatBuffer[] cl_errorBuffers = new CLFloatBuffer[populationSize];
            int[][] usedColorsArray = new int[populationSize][];
            float[][] errorArray = new float[populationSize][];
            for(int i=0; i<populationSize; i++)
            {
                usedColorBuffers[i] = ByteBuffer.allocateDirect(nbOfColors*4).order(context.getByteOrder()).asIntBuffer();
                cl_usedColorBuffers[i] = context.createIntBuffer(CLMem.Usage.Output, usedColorBuffers[i], false);
                errorBuffers[i] = ByteBuffer.allocateDirect(inlineScielabOriginal.length*4).order(context.getByteOrder()).asFloatBuffer();
                cl_errorBuffers[i] = context.createFloatBuffer(CLMem.Usage.Output, errorBuffers[i], false);
                usedColorsArray[i] = new int[nbOfColors];
                errorArray[i] = new float[inlinergbOriginal.length/4];
            }

            //IntBuffer usedColorBuffer = ByteBuffer.allocateDirect(nbOfColors*4).order(context.getByteOrder()).asIntBuffer();
            //CLIntBuffer cl_usedColorBuffer = context.createIntBuffer(CLMem.Usage.Output, usedColorBuffer, false);
            //FloatBuffer errorBuffer = ByteBuffer.allocateDirect(inlineScielabOriginal.length*4).order(context.getByteOrder()).asFloatBuffer();
            //CLFloatBuffer cl_errorBuffer = context.createFloatBuffer(CLMem.Usage.Output, errorBuffer, false);

            //Input buffers
            CLFloatBuffer cl_comparisonBuffer = context.createFloatBuffer(CLMem.Usage.Input,inlineScielabOriginal.length);
            CLFloatBuffer cl_rgbBuffer = context.createFloatBuffer(CLMem.Usage.Input,inlinergbOriginal.length);
            CLFloatBuffer cl_colorBuffer = context.createFloatBuffer(CLMem.Usage.Input, colors[0].length);

            //Filter buffers
            CLFloatBuffer cl_filterBuffer = context.createFloatBuffer(CLMem.Usage.Input,filterLength);
            CLFloatBuffer cl_filterBuffer2 = context.createFloatBuffer(CLMem.Usage.Input,filterLength);
            CLFloatBuffer cl_filterBuffer3 = context.createFloatBuffer(CLMem.Usage.Input,filterLength/4);
            CLFloatBuffer cl_absfilterBuffer = context.createFloatBuffer(CLMem.Usage.Input,filterLength/4);

            //Middle buffers
            CLFloatBuffer cl_OppBuffer = context.createFloatBuffer(CLMem.Usage.InputOutput, inlineScielabOriginal.length);
            CLFloatBuffer cl_temp1Buffer = context.createFloatBuffer(CLMem.Usage.InputOutput, inlineScielabOriginal.length);
            CLFloatBuffer cl_temp2Buffer = context.createFloatBuffer(CLMem.Usage.InputOutput, inlineScielabOriginal.length);
            CLFloatBuffer cl_temp3Buffer = context.createFloatBuffer(CLMem.Usage.InputOutput, inlineScielabOriginal.length/4);
            CLFloatBuffer cl_convBuffer = context.createFloatBuffer(CLMem.Usage.InputOutput, inlineScielabOriginal.length);
            CLFloatBuffer cl_labBuffer = context.createFloatBuffer(CLMem.Usage.InputOutput, inlineScielabOriginal.length);

            //BUFFER LOADING

            //Input buffers
            loadcomp = loadGPUBuffer(cl_comparisonBuffer,inlineScielabOriginal);
            loadrgb = loadGPUBuffer(cl_rgbBuffer, inlinergbOriginal);

            //Filter buffers
            loadfilter1 = loadGPUBuffer(cl_filterBuffer, filters4[0]);
            loadfilter2 = loadGPUBuffer(cl_filterBuffer2, filters4[1]);
            loadfilter3 = loadGPUBuffer(cl_filterBuffer3, filter3);
            loadabs = loadGPUBuffer(cl_absfilterBuffer, absfilter3);

            //ARGUMENTS DEFINITION FOR CONSTANT KERNELS
            Quantize2OppKernel.setArgs(cl_rgbBuffer, cl_colorBuffer, nbOfColors, cl_usedColorBuffers[0], cl_OppBuffer);
            DeltaEKernel.setArgs(cl_comparisonBuffer, cl_labBuffer, cl_errorBuffers[0]);
            Opp2LABKernel.setArgs(cl_convBuffer, illuminant[0], illuminant[1], illuminant[2], cl_labBuffer);
            convolutionScielabTemp.setArgs(cl_OppBuffer,cl_filterBuffer,cl_filterBuffer2, cl_filterBuffer3,filterHalfWidth,w,h, cl_temp1Buffer,  cl_temp2Buffer, cl_temp3Buffer);
            convolutionScielabEnd.setArgs(cl_temp1Buffer, cl_temp2Buffer, cl_temp3Buffer,cl_filterBuffer,cl_filterBuffer2, cl_absfilterBuffer,filterHalfWidth,h,w, cl_convBuffer);

            CLEvent.waitFor(loadcomp, loadrgb, loadfilter1, loadfilter2, loadfilter3, loadabs);

            //bestError = currentError = computeQuantizationError(cl_filterBuffer, cl_filterBuffer2, cl_filterBuffer3, cl_absfilterBuffer, cl_OppBuffer, cl_temp1Buffer,cl_temp2Buffer,cl_temp3Buffer, cl_convBuffer, cl_usedColorBuffer, usedColorBuffer,usedColors, cl_errorBuffer, errorBuffer, errorArray, w, h, workSize, filterHalfWidth, simulatedAnnealing);
            double[] currentErrors = computeQuantizationErrorPopulation(populationSize, initialUsedColors, cl_colorBuffer,colors, cl_usedColorBuffers, usedColorBuffers,usedColorsArray,cl_errorBuffers,errorBuffers,errorArray,workSize, simulatedAnnealing, events );
            int min = argmin(currentErrors);
            bestError = currentErrors[min];
            System.arraycopy(colors[min], 0, bestColors, 0,bestColors.length);
            //MAIN LOOP
            long start = System.currentTimeMillis();
            int maxiter = simulatedAnnealing.getImax();
            for(int ite=1; ite<=maxiter;ite++)
            {
                //usedColorBuffer.rewind();
                //usedColorBuffer.put(initialUsedColors);
                //usedColorBuffer.rewind();
                //CLEvent event = cl_usedColorBuffer.write(queue,usedColorBuffer, false);
                simulatedAnnealing.reduceTemperatureIfNecessary(ite);
                for(int j=0; j<populationSize; j++)
                {
                    simulatedAnnealing.generateNeighboringColors(colors[j], currentColors[j],nbOfColors, ite);
                }

                //loadGPUBuffer(cl_colorBuffer, currentColors, event).waitFor();
                //error = computeQuantizationError(cl_filterBuffer, cl_filterBuffer2, cl_filterBuffer3, cl_absfilterBuffer, cl_OppBuffer, cl_temp1Buffer,cl_temp2Buffer,cl_temp3Buffer, cl_convBuffer, cl_usedColorBuffer, usedColorBuffer,usedColors, cl_errorBuffer, errorBuffer, errorArray, w, h, workSize, filterHalfWidth, simulatedAnnealing);
                double[] errors = computeQuantizationErrorPopulation(populationSize, initialUsedColors, cl_colorBuffer,currentColors, cl_usedColorBuffers, usedColorBuffers,usedColorsArray,cl_errorBuffers,errorBuffers,errorArray,workSize, simulatedAnnealing, events );
                for(int i=0; i<populationSize; i++)
                {
                    if(simulatedAnnealing.isAccepted(errors[i]-currentErrors[i]))
                    {
                        currentErrors[i] = errors[i];
                        System.arraycopy(currentColors[i], 0, colors[i],0, currentColors[i].length);
                        if(currentErrors[i] < bestError)
                        {
                            bestError = currentErrors[i];
                            System.arraycopy(currentColors[i], 0, bestColors,0, currentColors[i].length);
                            System.out.println("Best Error :" + bestError);
                        }
                    }
                }

                if(simulatedAnnealing.getPlugin().isStopFlag())
                {
                    break;
                }
                if(ite % 10 == 0)
                {
                    long elapsed = System.currentTimeMillis();
                    long restant=(long)((((elapsed-start)*1.0)/ite)*(maxiter-ite));
                    String tpsRestant = (restant/60000 > 0 ? restant/60000 + "m" : "") + ((restant%60000)/1000 + "s") + " restant";
                    simulatedAnnealing.getPlugin().updateProgressBar(ite+"/"+maxiter + " : " + tpsRestant,(ite*1.0)/maxiter);
                    for(double c : currentErrors)
                    {
                        System.out.printf("%.5f",c);
                        System.out.print('\t');
                    }
                    System.out.println();
                }
            }

            //Releasing the buffers
            cl_absfilterBuffer.release();
            cl_colorBuffer.release();
            cl_comparisonBuffer.release();
            cl_convBuffer.release();
            for(CLFloatBuffer b : cl_errorBuffers)
                b.release();
            cl_filterBuffer.release();
            cl_filterBuffer2.release();
            cl_filterBuffer3.release();
            cl_OppBuffer.release();
            cl_rgbBuffer.release();
            cl_temp1Buffer.release();
            cl_temp2Buffer.release();
            cl_temp3Buffer.release();
            cl_labBuffer.release();
            for(CLIntBuffer b : cl_usedColorBuffers)
                b.release();
        }
        return bestColors;
    }

    private double computeQuantizationError(CLFloatBuffer filter1, CLFloatBuffer filter2, CLFloatBuffer filter3, CLFloatBuffer absfilter, CLFloatBuffer opp, CLFloatBuffer temp1, CLFloatBuffer temp2, CLFloatBuffer temp3, CLFloatBuffer conv, CLIntBuffer usedColorBuffer, IntBuffer usedColors,int[] usedColorsArray, CLFloatBuffer errorBuffer, FloatBuffer errors, float[] errorArray, int w, int h, int[] worksize, int filterHalfWidth, SWASA swasa) {
        CLEvent event;
        //Quantification
        event = Quantize2OppKernel.enqueueNDRange(queue, worksize);
        usedColorBuffer.read(queue, usedColors, false, event);

        //Convolution horizontale
        event = convolutionScielabTemp.enqueueNDRange(queue, worksize, event);

        //Convolution verticale
        event = convolutionScielabEnd.enqueueNDRange(queue, worksize, event);

        //Conversion en lab
        event = Opp2LABKernel.enqueueNDRange(queue, worksize, event);

        //Calcul de l'erreur
        event = DeltaEKernel.enqueueNDRange(queue, worksize, event);
        queue.finish();
        errorBuffer.read(queue,errors, true, event);
        errors.rewind();
        errors.get(errorArray);
        usedColors.rewind();
        usedColors.get(usedColorsArray);

        return averageArray(errorArray)/3 + swasa.computePenalty(usedColorsArray);
    }

    private double[] computeQuantizationErrorPopulation(int populationSize, int[] cleanUsedColors, CLFloatBuffer cl_colors, float[][] colors, CLIntBuffer[] usedColorBuffer, IntBuffer[] usedColors,int[][] usedColorsArray, CLFloatBuffer[] errorBuffer, FloatBuffer[] errors, float[][] errorArray, int[] worksize, SWASA swasa, CLEvent[][] events) {
        /* Le deuxieme indice des events est l'étape, entre parentheses est la dépendance de population et d'étape :
            0 : Reinitialisation du buffer des couleurs utilisées (i-1, 3)
            1 : Ecriture des couleurs (i-1, 2)
            2 : Quantification (i, 0)(i, 1), (i-1 , 4)
            3 : Lecture des couleurs utilisées (i, 2)
            4 : Scielab horizontal (i, 2)(i-1, 5)
            5 : Scielab vertical (i, 4)(i-1, 6)
            6 : Conversion en lab (i, 5)(i-1, 7)
            7 : Calcul de l'erreur (i, 6)
            8 : Lecture de l'erreur (i, 7)
         */
        double[] results = new double[populationSize];
        Thread[] threads = new Thread[populationSize];

        for(int i=0; i<populationSize; i++)
        {
            if(i > 0)
            {
                //Reinitialisation du buffer
                events[i][0] = usedColorBuffer[i].write(queue,usedColors[i], false, events[i-1][3]);

                //Ecriture des couleurs dans le buffer
                events[i][1] = loadGPUBuffer(cl_colors, colors[i], events[i-1][2]);

                //Quantification
                Quantize2OppKernel.setArg(3,usedColorBuffer[i]);
                events[i][2] = Quantize2OppKernel.enqueueNDRange(queue, worksize, events[i-1][4], events[i][0], events[i][1]);

                //Lecture des couleurs utilisées
                events[i][3] = usedColorBuffer[i].read(queue, usedColors[i], false, events[i][2]);

                //Scielab horizontal
                events[i][4] = convolutionScielabTemp.enqueueNDRange(queue, worksize, events[i][2], events[i-1][5]);

                //Scielab vertical
                events[i][5] = convolutionScielabEnd.enqueueNDRange(queue, worksize, events[i][4], events[i-1][6]);

                //Scielab horizontal
                events[i][6] = Opp2LABKernel.enqueueNDRange(queue, worksize, events[i][5], events[i-1][7]);

                //Calcul de l'erreur
                DeltaEKernel.setArg(2, errorBuffer[i]);
                events[i][7] = DeltaEKernel.enqueueNDRange(queue, worksize, events[i][6]);

                //Lecture de l'erreur
                errors[i].rewind();
                events[i][8] = errorBuffer[i].read(queue,errors[i], false, events[i][7]);

            }else
            {
                //Reinitialisation du buffer
                events[i][0] = usedColorBuffer[i].write(queue,usedColors[i], false);

                //Ecriture des couleurs dans le buffer
                events[i][1] = loadGPUBuffer(cl_colors, colors[i]);

                //Quantification
                Quantize2OppKernel.setArg(3,usedColorBuffer[i]);
                events[i][2] = Quantize2OppKernel.enqueueNDRange(queue, worksize, events[i][0], events[i][1]);

                //Lecture des couleurs utilisées
                events[i][3] = usedColorBuffer[i].read(queue, usedColors[i], false, events[i][2]);

                //Scielab horizontal
                events[i][4] = convolutionScielabTemp.enqueueNDRange(queue, worksize, events[i][2]);

                //Scielab vertical
                events[i][5] = convolutionScielabEnd.enqueueNDRange(queue, worksize, events[i][4]);

                //Scielab horizontal
                events[i][6] = Opp2LABKernel.enqueueNDRange(queue, worksize, events[i][5]);

                //Calcul de l'erreur
                DeltaEKernel.setArg(2, errorBuffer[i]);
                events[i][7] = DeltaEKernel.enqueueNDRange(queue, worksize, events[i][6]);

                //Lecture de l'erreur
                events[i][8] = errorBuffer[i].read(queue,errors[i], false, events[i][7]);
            }

            int finalI = i;
            threads[i] = new Thread(()->{
                events[finalI][3].waitFor();
                usedColors[finalI].rewind();
                usedColors[finalI].get(usedColorsArray[finalI]).rewind();
                usedColors[finalI].put(cleanUsedColors).rewind();

                events[finalI][8].waitFor();
                errors[finalI].rewind();
                errors[finalI].get(errorArray[finalI]).rewind();

                results[finalI] = averageArray(errorArray[finalI])/3 + swasa.computePenalty(usedColorsArray[finalI]);
            });
            threads[i].start();
        }

        queue.finish();
        for(Thread t : threads)
        {
            try {
                t.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        return results;
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
        filter3 = new float[filters[0][2].length];
        for(int j=0; j < filters[0][2].length; j++)
        {
            off = j<<2;
            filters4[2][off] = filters[0][2][j];
            filters4[2][off+1] = 0.0f;
            filters4[2][off+2] = 0.0f;
            filters4[2][off+3] = 0.0f;
            filter3[j] = filters[0][2][j];
        }

        absfilter3 = new float[absfilters.length];
        for(int j=0; j < absfilters.length; j++)
        {
            off = j<<2;
            absfilters4[off] = absfilters[j];
            absfilters4[off+1] = 0.0f;
            absfilters4[off+2] = 0.0f;
            absfilters4[off+3] = 0.0f;
            absfilter3[j] = absfilters[j];
        }


        openCLFiltersReady=true;
    }

    public static int argmin(double[] arr)
    {
        int min =0;
        double score = arr[0];
        for(int i=1; i<arr.length; i++)
        {
            if(score > arr[i])
            {
                min =i;
                score = arr[i];
            }
        }
        return min;
    }
}
