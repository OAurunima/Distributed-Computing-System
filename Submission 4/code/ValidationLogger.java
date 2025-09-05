import peersim.config.Configuration;
import peersim.core.CommonState;
import peersim.core.Control;
import peersim.core.Network;
import peersim.core.Node;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Validation Logger for Enhanced Energy-Aware Protocol
 * Provides comprehensive statistical analysis and validation metrics
 */
public class ValidationLogger implements Control {
    private static final String PAR_PROTOCOL = "protocol";
    private static final String PAR_OUTPUT_FILE = "output_file";
    private static final String PAR_STATS_FILE = "stats_file";
    
    private final String protocolName;
    private final String outputFile;
    private final String statsFile;
    private FileWriter csvWriter;
    private FileWriter statsWriter;
    private boolean headersWritten = false;
    
    // Statistical tracking
    private List<Double> allScores = new ArrayList<>();
    private List<Double> allSpeeds = new ArrayList<>();
    private List<Double> allEnergy = new ArrayList<>();
    private int cycleCount = 0;

    public ValidationLogger(String prefix) {
        this.protocolName = Configuration.getString(prefix + "." + PAR_PROTOCOL);
        this.outputFile = Configuration.getString(prefix + "." + PAR_OUTPUT_FILE, "validation_metrics.csv");
        this.statsFile = Configuration.getString(prefix + "." + PAR_STATS_FILE, "statistical_analysis.csv");
        
        try {
            csvWriter = new FileWriter(outputFile);
            statsWriter = new FileWriter(statsFile);
        } catch (IOException e) {
            throw new RuntimeException("Cannot create output files", e);
        }
    }

    @Override
    public boolean execute() {
        try {
            cycleCount++;
            logCycleData();
            
            // Perform statistical analysis every 10 cycles
            if (cycleCount % 10 == 0) {
                performStatisticalAnalysis();
            }
            
            return false;
        } catch (IOException e) {
            throw new RuntimeException("Error writing to output files", e);
        }
    }
    
    private void logCycleData() throws IOException {
        if (!headersWritten) {
            writeHeaders();
            headersWritten = true;
        }
        
        List<NodeMetrics> nodeMetrics = collectNodeMetrics();
        
        // Log individual node data
        for (NodeMetrics metrics : nodeMetrics) {
            csvWriter.write(String.format("%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%s%n",
                CommonState.getTime(), metrics.nodeId,
                metrics.energyConsumption, metrics.uploadSpeed, metrics.score,
                metrics.normalizedScore, metrics.performanceIndex,
                metrics.stabilityFactor, metrics.confidenceInterval,
                metrics.performanceCategory, metrics.isStable));
        }
        
        csvWriter.flush();
    }
    
    private void writeHeaders() throws IOException {
        csvWriter.write("cycle,node_id,energy_consumption,upload_speed,score," +
                       "normalized_score,performance_index,stability_factor," +
                       "confidence_interval,performance_category,is_stable\n");
    }
    
    private List<NodeMetrics> collectNodeMetrics() {
        List<NodeMetrics> metrics = new ArrayList<>();
        int protocolId = CommonState.getPid(protocolName);
        
        allScores.clear();
        allSpeeds.clear();
        allEnergy.clear();
        
        for (int i = 0; i < Network.size(); i++) {
            Node node = Network.get(i);
            EnhancedEnergyProtocol protocol = (EnhancedEnergyProtocol) node.getProtocol(protocolId);
            
            NodeMetrics nm = new NodeMetrics();
            nm.nodeId = i;
            nm.energyConsumption = protocol.energyConsumption;
            nm.uploadSpeed = protocol.uploadSpeed;
            nm.score = protocol.score;
            nm.normalizedScore = protocol.normalizedScore;
            nm.performanceIndex = protocol.performanceIndex;
            nm.stabilityFactor = protocol.stabilityFactor;
            nm.confidenceInterval = protocol.getConfidenceInterval();
            nm.performanceCategory = protocol.getPerformanceCategory();
            nm.isStable = protocol.isPerformanceStable();
            
            metrics.add(nm);
            
            // Collect for statistical analysis
            allScores.add(protocol.score);
            allSpeeds.add(protocol.uploadSpeed);
            allEnergy.add(protocol.energyConsumption);
        }
        
        // Update efficiency ranks based on global distribution
        updateEfficiencyRanks(metrics);
        
        return metrics;
    }
    
    private void updateEfficiencyRanks(List<NodeMetrics> metrics) {
        // Sort by score for ranking
        List<NodeMetrics> sorted = new ArrayList<>(metrics);
        sorted.sort(Comparator.comparingDouble((NodeMetrics nm) -> nm.score).reversed());
        
        // Assign percentile ranks
        for (int i = 0; i < sorted.size(); i++) {
            int protocolId = CommonState.getPid(protocolName);
            Node node = Network.get(sorted.get(i).nodeId);
            EnhancedEnergyProtocol protocol = (EnhancedEnergyProtocol) node.getProtocol(protocolId);
            protocol.efficiencyRank = 1.0 - (double) i / sorted.size(); // Higher score = higher rank
        }
    }
    
    private void performStatisticalAnalysis() throws IOException {
        if (allScores.isEmpty()) return;
        
        StatisticalSummary scoreSummary = calculateStatistics(allScores);
        StatisticalSummary speedSummary = calculateStatistics(allSpeeds);
        StatisticalSummary energySummary = calculateStatistics(allEnergy);
        
        // Write statistical summary
        if (cycleCount == 10) { // Write header only once
            statsWriter.write("cycle,metric,mean,median,std_dev,min,max,q25,q75," +
                             "skewness,kurtosis,coefficient_variation\n");
        }
        
        writeStatsSummary("score", scoreSummary);
        writeStatsSummary("speed", speedSummary);
        writeStatsSummary("energy", energySummary);
        
        // Correlation analysis
        double correlation = calculateCorrelation(allSpeeds, allEnergy);
        statsWriter.write(String.format("%d,speed_energy_correlation,%.6f,0,0,0,0,0,0,0,0,0%n",
                         CommonState.getTime(), correlation));
        
        statsWriter.flush();
    }
    
    private void writeStatsSummary(String metricName, StatisticalSummary summary) throws IOException {
        statsWriter.write(String.format("%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f%n",
            CommonState.getTime(), metricName,
            summary.mean, summary.median, summary.stdDev,
            summary.min, summary.max, summary.q25, summary.q75,
            summary.skewness, summary.kurtosis, summary.coefficientOfVariation));
    }
    
    private StatisticalSummary calculateStatistics(List<Double> data) {
        if (data.isEmpty()) return new StatisticalSummary();
        
        List<Double> sorted = new ArrayList<>(data);
        Collections.sort(sorted);
        
        StatisticalSummary summary = new StatisticalSummary();
        summary.min = sorted.get(0);
        summary.max = sorted.get(sorted.size() - 1);
        summary.median = getPercentile(sorted, 0.5);
        summary.q25 = getPercentile(sorted, 0.25);
        summary.q75 = getPercentile(sorted, 0.75);
        
        // Calculate mean
        double sum = data.stream().mapToDouble(Double::doubleValue).sum();
        summary.mean = sum / data.size();
        
        // Calculate standard deviation
        double variance = data.stream()
            .mapToDouble(x -> Math.pow(x - summary.mean, 2))
            .sum() / data.size();
        summary.stdDev = Math.sqrt(variance);
        
        // Calculate coefficient of variation
        summary.coefficientOfVariation = summary.stdDev / summary.mean;
        
        // Calculate skewness and kurtosis (simplified versions)
        double moment3 = data.stream()
            .mapToDouble(x -> Math.pow((x - summary.mean) / summary.stdDev, 3))
            .sum() / data.size();
        summary.skewness = moment3;
        
        double moment4 = data.stream()
            .mapToDouble(x -> Math.pow((x - summary.mean) / summary.stdDev, 4))
            .sum() / data.size();
        summary.kurtosis = moment4 - 3; // Excess kurtosis
        
        return summary;
    }
    
    private double getPercentile(List<Double> sorted, double percentile) {
        int index = (int) Math.ceil(percentile * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(index, sorted.size() - 1)));
    }
    
    private double calculateCorrelation(List<Double> x, List<Double> y) {
        if (x.size() != y.size() || x.isEmpty()) return 0.0;
        
        double meanX = x.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double meanY = y.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        double numerator = 0.0;
        double sumSqX = 0.0;
        double sumSqY = 0.0;
        
        for (int i = 0; i < x.size(); i++) {
            double dx = x.get(i) - meanX;
            double dy = y.get(i) - meanY;
            numerator += dx * dy;
            sumSqX += dx * dx;
            sumSqY += dy * dy;
        }
        
        double denominator = Math.sqrt(sumSqX * sumSqY);
        return denominator == 0.0 ? 0.0 : numerator / denominator;
    }
    
    // Helper classes
    private static class NodeMetrics {
        int nodeId;
        double energyConsumption;
        double uploadSpeed;
        double score;
        double normalizedScore;
        double performanceIndex;
        double stabilityFactor;
        double confidenceInterval;
        String performanceCategory;
        boolean isStable;
    }
    
    private static class StatisticalSummary {
        double mean, median, stdDev, min, max, q25, q75;
        double skewness, kurtosis, coefficientOfVariation;
    }
}