package plugins.dbrasseur.hybridquantization;

public class SWASA {

    private int population,imax,iTc;
    private float delta,t0,alpha,s0,beta;

    private float temperature;
    private float stepWidth;

    private HybridQuantization plugin;

    public SWASA(int population, int imax, int iTc, float delta, float t0, float alpha, float s0, float beta, HybridQuantization HQ) {
        plugin = HQ;
        this.population = population;
        this.imax = imax;
        this.iTc = iTc;
        this.delta = delta;
        this.t0 = t0;
        this.alpha = alpha;
        this.s0 = s0;
        this.beta = beta;

        reset();
    }

    public void reset()
    {
        temperature=t0;
        stepWidth=s0;
    }

    public int getImax() {
        return imax;
    }

    public float[] generateRandomColors(int numberOfColors)
    {
        float[] colors = new float[4*numberOfColors];
        for(int i=0; i<numberOfColors;i++)
        {
            int offset = i<<2;
            colors[offset] = icy.util.Random.nextFloat();
            colors[offset+1] = icy.util.Random.nextFloat();
            colors[offset+2] = icy.util.Random.nextFloat();
            colors[offset+3] = 0.0f;
        }
        return colors;
    }

    public boolean isAccepted(double deltaE)
    {
        return deltaE <= 0 || acceptanceProbability(deltaE*256) > icy.util.Random.nextDouble();
    }

    private double acceptanceProbability(double deltaE)
    {
        return Math.exp(-deltaE/temperature);
    }

    private float maxStepWidth(int i)
    {
        return (float)(2*s0/(1+Math.exp(beta*i/imax)));
    }

    public double computePenalty(int[] usedColors) {
        double penalty = 0;
        for(int c : usedColors)
        {
            if(c == 0)
                penalty+=delta;
        }
        return penalty;
    }

    public void reduceTemperatureIfNecessary(int iteration) {
        if(iteration % iTc == 0)
        {
            temperature*=alpha;
        }
    }

    public void generateNeighboringColors(float[] colors, float[] nextColors, int numberOfColors,int iteration) {
        float actualMaxStepWidth = maxStepWidth(iteration)/256.0f;
        for(int i=0; i<numberOfColors;i++)
        {
            int offset = i<<2;
            nextColors[offset] = clamp(colors[offset]+(icy.util.Random.nextFloat()*2 -1)*actualMaxStepWidth, 0, 1);
            nextColors[offset+1] = clamp(colors[offset+1]+(icy.util.Random.nextFloat()*2 -1)*actualMaxStepWidth, 0, 1);
            nextColors[offset+2] = clamp(colors[offset+2]+(icy.util.Random.nextFloat()*2 -1)*actualMaxStepWidth, 0, 1);
            nextColors[offset+3] = 0.0f;
        }
    }

    public static float clamp(float value, float min, float max)
    {
        return value > min ? value > max ? max : value : min;
    }
}
