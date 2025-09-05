import peersim.cdsim.CDProtocol;
import peersim.core.Node;
import peersim.config.Configuration;
import peersim.core.CommonState;
import java.util.Random;

/**
 * Enhanced Energy-Aware Protocol with Validation Techniques
 * Includes statistical analysis, performance validation, and comparative metrics
 */
public class EnhancedEnergyProtocol implements CDProtocol {
    private static final String PAR_BASE_ENERGY = "base_energy";      
    private static final String PAR_ENERGY_VAR = "energy_variance";   
    private static final String PAR_SPEED_MIN = "speed_min";          
    private static final String PAR_SPEED_MAX = "speed_max";          
    private static final String PAR_EFFICIENCY_WEIGHT = "efficiency_weight";
    
    private final double baseEnergy;
    private final double energyVariance;
    private final double speedMin;
    private final double speedMax;
    private final double efficiencyWeight;
    private final Random rng;

    // Core metrics
    public double energyConsumption;
    public double uploadSpeed;
    public double score;
    
    // Enhanced validation metrics
    public double normalizedScore;
    public double performanceIndex;
    public double efficiencyRank;
    public double stabilityFactor;
    
    // Historical data for validation
    private double[] recentScores = new double[10];
    private int historyIndex = 0;
    private boolean historyFull = false;

    public EnhancedEnergyProtocol(String prefix) {
        this.baseEnergy = Configuration.getDouble(prefix + "." + PAR_BASE_ENERGY, 60.0);
        this.energyVariance = Configuration.getDouble(prefix + "." + PAR_ENERGY_VAR, 15.0);
        this.speedMin = Configuration.getDouble(prefix + "." + PAR_SPEED_MIN, 5.0);
        this.speedMax = Configuration.getDouble(prefix + "." + PAR_SPEED_MAX, 25.0);
        this.efficiencyWeight = Configuration.getDouble(prefix + "." + PAR_EFFICIENCY_WEIGHT, 0.7);
        this.rng = new Random(CommonState.r.nextLong());
    }

    @Override
    public void nextCycle(Node node, int pid) {
        // Generate realistic upload speed with temporal correlation
        uploadSpeed = generateRealisticSpeed();
        
        // Generate energy consumption with load-dependent model
        energyConsumption = generateEnergyConsumption();
        
        // Calculate basic energy-aware score
        score = uploadSpeed / energyConsumption;
        
        // Update historical data
        updateHistory();
        
        // Calculate enhanced validation metrics
        calculateValidationMetrics();
    }
    
    private double generateRealisticSpeed() {
        // Simulate realistic network conditions with temporal correlation
        double baseSpeed = speedMin + (speedMax - speedMin) * rng.nextDouble();
        
        // Add network congestion effects
        double congestionFactor = 0.8 + 0.4 * rng.nextDouble();
        
        // Add temporal stability (slight correlation with previous values)
        if (historyFull && recentScores.length > 0) {
            double previousInfluence = 0.1 * (uploadSpeed * 0.5);
            baseSpeed += previousInfluence;
        }
        
        return Math.max(1.0, baseSpeed * congestionFactor);
    }
    
    private double generateEnergyConsumption() {
        // Load-dependent energy model: base + variable based on performance
        double variableEnergy = uploadSpeed * (0.5 + 0.3 * rng.nextDouble());
        double baseConsumption = baseEnergy + rng.nextGaussian() * energyVariance;
        
        return Math.max(20.0, baseConsumption + variableEnergy);
    }
    
    private void updateHistory() {
        recentScores[historyIndex] = score;
        historyIndex = (historyIndex + 1) % recentScores.length;
        if (!historyFull && historyIndex == 0) {
            historyFull = true;
        }
    }
    
    private void calculateValidationMetrics() {
        // Normalized score (0-1 scale based on theoretical max efficiency)
        double maxPossibleScore = speedMax / 20.0; // Theoretical minimum energy
        normalizedScore = Math.min(1.0, score / maxPossibleScore);
        
        // Performance index combining speed and efficiency
        performanceIndex = (efficiencyWeight * normalizedScore) + 
                          ((1 - efficiencyWeight) * (uploadSpeed / speedMax));
        
        // Stability factor based on recent score variance
        stabilityFactor = calculateStabilityFactor();
        
        // Efficiency rank (percentile-based ranking placeholder)
        efficiencyRank = normalizedScore; // Will be updated by global ranking in analyzer
    }
    
    private double calculateStabilityFactor() {
        if (!historyFull) return 0.5; // Default for insufficient data
        
        double mean = 0.0;
        int count = historyFull ? recentScores.length : historyIndex;
        
        for (int i = 0; i < count; i++) {
            mean += recentScores[i];
        }
        mean /= count;
        
        double variance = 0.0;
        for (int i = 0; i < count; i++) {
            variance += Math.pow(recentScores[i] - mean, 2);
        }
        variance /= count;
        
        // Convert variance to stability (lower variance = higher stability)
        return 1.0 / (1.0 + variance);
    }
    
    // Validation methods for statistical analysis
    public double getConfidenceInterval() {
        if (!historyFull) return 0.0;
        
        double mean = 0.0;
        for (double s : recentScores) {
            mean += s;
        }
        mean /= recentScores.length;
        
        double stdDev = 0.0;
        for (double s : recentScores) {
            stdDev += Math.pow(s - mean, 2);
        }
        stdDev = Math.sqrt(stdDev / recentScores.length);
        
        // 95% confidence interval
        return 1.96 * stdDev / Math.sqrt(recentScores.length);
    }
    
    public boolean isPerformanceStable() {
        return stabilityFactor > 0.7; // Threshold for stability
    }
    
    public String getPerformanceCategory() {
        if (performanceIndex > 0.8) return "High";
        else if (performanceIndex > 0.6) return "Medium";
        else if (performanceIndex > 0.4) return "Low";
        else return "Poor";
    }

    @Override
    public Object clone() {
        try {
            EnhancedEnergyProtocol copy = (EnhancedEnergyProtocol) super.clone();
            copy.recentScores = new double[10];
            copy.historyIndex = 0;
            copy.historyFull = false;
            return copy;
        } catch (CloneNotSupportedException e) {
            return null;
        }
    }
}