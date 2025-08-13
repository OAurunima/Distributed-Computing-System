import peersim.cdsim.CDProtocol;
import peersim.core.Node;
import peersim.config.Configuration;
import peersim.core.CommonState;
import java.util.Random;

/**
 * Per-peer state updated each cycle.
 * Fields: energyConsumption (W), uploadSpeed (MB/s), score = upload/energy.
 */
public class EnergyProtocol implements CDProtocol {
    private static final String PAR_BASE_ENERGY = "base_energy";      // e.g., 60
    private static final String PAR_ENERGY_PER_MB = "energy_per_mb";  // e.g., 0.8
    private static final String PAR_SPEED_MIN = "speed_min";          // e.g., 5
    private static final String PAR_SPEED_MAX = "speed_max";          // e.g., 25

    private final double baseEnergy;
    private final double energyPerMB;
    private final double speedMin;
    private final double speedMax;
    private final Random rng;

    // Per-node state
    public double energyConsumption;
    public double uploadSpeed;
    public double score;

    public EnergyProtocol(String prefix) {
        this.baseEnergy   = Configuration.getDouble(prefix + "." + PAR_BASE_ENERGY, 60.0);
        this.energyPerMB  = Configuration.getDouble(prefix + "." + PAR_ENERGY_PER_MB, 0.8);
        this.speedMin     = Configuration.getDouble(prefix + "." + PAR_SPEED_MIN, 5.0);
        this.speedMax     = Configuration.getDouble(prefix + "." + PAR_SPEED_MAX, 25.0);
        this.rng          = new Random(CommonState.r.nextLong());
    }

    @Override
    public void nextCycle(Node node, int pid) {
        // synthetic upload speed per cycle
        uploadSpeed = speedMin + (speedMax - speedMin) * rng.nextDouble();
        // simple energy model
        energyConsumption = baseEnergy + energyPerMB * uploadSpeed;
        score = uploadSpeed / energyConsumption;
        // nothing else here; logging is done by a Control
    }

    @Override
    public Object clone() {
        try { return super.clone(); } catch (Exception e) { return null; }
    }
}
