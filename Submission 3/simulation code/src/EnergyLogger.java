import peersim.core.Control;
import peersim.core.Network;
import peersim.core.Node;
import peersim.config.Configuration;
import java.io.FileWriter;
import java.io.PrintWriter;

/** Dumps per-peer CSV each cycle: peer_id,cycle,energy,upload,score */
public class EnergyLogger implements Control {
    private static final String PAR_PID = "protocol";
    private static final String PAR_OUT = "out";
    private final int pid;
    private final String outPath;

    public EnergyLogger(String prefix) {
        this.pid = Configuration.getPid(prefix + "." + PAR_PID);
        this.outPath = Configuration.getString(prefix + "." + PAR_OUT, "peer_metrics.csv");
    }

    @Override
    public boolean execute() {
        int cycle = peersim.core.CommonState.getIntTime();
        try (PrintWriter pw = new PrintWriter(new FileWriter(outPath, true))) {
            if (cycle == 0) pw.println("peer_id,cycle,energy_consumption_w,upload_speed_mbps,score");
            for (int i = 0; i < Network.size(); i++) {
                Node n = Network.get(i);
                EnergyProtocol ep = (EnergyProtocol) n.getProtocol(pid);
                pw.printf("%d,%d,%.6f,%.6f,%.6f%n", (int)n.getID(), cycle,
                        ep.energyConsumption, ep.uploadSpeed, ep.score);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return false; // keep going
    }
}