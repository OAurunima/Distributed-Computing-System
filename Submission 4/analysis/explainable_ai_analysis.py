#!/usr/bin/env python3
"""
Explainable AI Components for Energy-Aware Peer Selection
Provides interpretable analysis of peer selection decisions and model validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ExplainableEnergyAnalyzer:
    def __init__(self, data_file=None):
        """Initialize the analyzer with simulation data"""
        self.data = None
        self.models = {}
        self.feature_names = []
        self.target_name = 'score'
        self.scaler = StandardScaler()
        
        if data_file:
            self.load_data(data_file)
    
    def load_data(self, data_file):
        """Load and preprocess simulation data"""
        try:
            self.data = pd.read_csv(data_file)
            print(f"Loaded {len(self.data)} records from {data_file}")
            
            # Create derived features for better explainability
            self.create_derived_features()
            self.prepare_features()
            
        except FileNotFoundError:
            print(f"Data file {data_file} not found. Generating synthetic data...")
            self.generate_synthetic_data()
    
    def generate_synthetic_data(self, n_samples=5000):
        """Generate realistic synthetic data for demonstration"""
        np.random.seed(42)
        
        # Generate base features
        cycles = np.random.randint(1, 101, n_samples)
        node_ids = np.random.randint(0, 300, n_samples)
        
        # Generate correlated features
        base_energy = 60 + np.random.normal(0, 15, n_samples)
        upload_speed = 5 + 20 * np.random.random(n_samples)
        
        # Add realistic correlations
        energy_consumption = np.maximum(20, base_energy + 0.3 * upload_speed + 
                                      np.random.normal(0, 5, n_samples))
        
        # Calculate scores and derived metrics
        scores = upload_speed / energy_consumption
        normalized_scores = scores / np.max(scores)
        performance_index = 0.7 * normalized_scores + 0.3 * (upload_speed / 25)
        stability_factor = 0.5 + 0.5 * np.random.random(n_samples)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'cycle': cycles,
            'node_id': node_ids,
            'energy_consumption': energy_consumption,
            'upload_speed': upload_speed,
            'score': scores,
            'normalized_score': normalized_scores,
            'performance_index': performance_index,
            'stability_factor': stability_factor
        })
        
        self.create_derived_features()
        self.prepare_features()
        print(f"Generated {len(self.data)} synthetic records")
    
    def create_derived_features(self):
        """Create interpretable derived features"""
        # Efficiency categories
        self.data['efficiency_ratio'] = self.data['upload_speed'] / self.data['energy_consumption']
        self.data['energy_per_mbps'] = self.data['energy_consumption'] / self.data['upload_speed']
        
        # Performance categories
        self.data['speed_category'] = pd.cut(self.data['upload_speed'], 
                                           bins=[0, 10, 15, 20, float('inf')], 
                                           labels=['Low', 'Medium', 'High', 'Very_High'])
        
        self.data['energy_category'] = pd.cut(self.data['energy_consumption'],
                                            bins=[0, 50, 70, 90, float('inf')],
                                            labels=['Efficient', 'Moderate', 'High', 'Excessive'])
        
        # Stability indicators
        node_stats = self.data.groupby('node_id')['score'].agg(['mean', 'std']).reset_index()
        node_stats['consistency'] = 1 / (1 + node_stats['std'])
        self.data = self.data.merge(node_stats[['node_id', 'consistency']], on='node_id', how='left')
        
        # Temporal features
        self.data['cycle_phase'] = self.data['cycle'] % 20  # Cyclical patterns
        self.data['is_early_cycle'] = (self.data['cycle'] <= 20).astype(int)
    
    def prepare_features(self):
        """Prepare features for ML models"""
        # Select numerical features for modeling
        feature_cols = ['energy_consumption', 'upload_speed', 'efficiency_ratio', 
                       'energy_per_mbps', 'consistency', 'cycle_phase', 'is_early_cycle']
        
        self.feature_names = feature_cols
        self.X = self.data[feature_cols].fillna(0)
        self.y = self.data[self.target_name]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)
    
    def train_interpretable_models(self):
        """Train various interpretable models"""
        models_config = {
            'linear': LinearRegression(),
            'decision_tree': DecisionTreeRegressor(max_depth=5, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        }
        
        print("Training interpretable models...")
        results = {}
        
        for name, model in models_config.items():
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Evaluate
            train_score = model.score(self.X_train, self.y_train)
            test_score = model.score(self.X_test, self.y_test)
            y_pred = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            self.models[name] = model
            results[name] = {
                'train_r2': train_score,
                'test_r2': test_score,
                'mse': mse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"{name.title()}: R² = {test_score:.4f}, MSE = {mse:.6f}, "
                  f"CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def explain_decision_tree(self, max_features=5):
        """Provide text explanation of decision tree rules"""
        if 'decision_tree' not in self.models:
            print("Decision tree model not found. Train models first.")
            return
        
        tree = self.models['decision_tree']
        tree_rules = export_text(tree, feature_names=self.feature_names, 
                                max_depth=3, spacing=2)
        
        print("=== DECISION TREE RULES FOR PEER SELECTION ===")
        print(tree_rules)
        
        return tree_rules
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance across models"""
        importance_data = {}
        
        # Get importance from tree-based models
        for name in ['decision_tree', 'random_forest', 'gradient_boosting']:
            if name in self.models:
                importance_data[name] = self.models[name].feature_importances_
        
        # Create importance DataFrame
        importance_df = pd.DataFrame(importance_data, index=self.feature_names)
        
        # Plot feature importance
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        importance_df.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('Feature Importance Across Models', fontsize=14, fontweight='bold', pad=20)
        axes[0].set_xlabel('Features', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Importance', fontsize=12, fontweight='bold')
        axes[0].legend(title='Models', title_fontsize=12, fontsize=10, loc='upper right')
        axes[0].tick_params(axis='x', rotation=0, labelsize=10)
        axes[0].tick_params(axis='y', labelsize=10)
        
        # Heatmap
        sns.heatmap(importance_df.T, annot=True, cmap='YlOrRd', ax=axes[1], 
                   cbar_kws={'label': 'Importance'}, fmt='.3f')
        axes[1].set_title('Feature Importance Heatmap', fontsize=14, fontweight='bold', pad=20)
        axes[1].set_xlabel('Features', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Models', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=0, labelsize=10)
        axes[1].tick_params(axis='y', rotation=0, labelsize=10)
        
        plt.tight_layout()
        plt.savefig('../images/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def shap_analysis(self):
        """Perform SHAP (SHapley Additive exPlanations) analysis"""
        if 'random_forest' not in self.models:
            print("Random forest model not found. Train models first.")
            return
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.models['random_forest'])
        shap_values = explainer.shap_values(self.X_test[:100])  # Sample for speed
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_test[:100], 
                         feature_names=self.feature_names, show=False)
        plt.title('SHAP Summary: Feature Impact on Peer Score')
        plt.tight_layout()
        plt.savefig('../images/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X_test[:100], 
                         feature_names=self.feature_names, 
                         plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig('../images/shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return shap_values
    
    def generate_peer_recommendations(self, threshold=0.15):
        """Generate explainable peer recommendations"""
        # Get predictions from best model
        best_model = self.models.get('random_forest', self.models.get('decision_tree'))
        if not best_model:
            print("No trained model available")
            return
        
        predictions = best_model.predict(self.X_test)
        
        # Classify peers
        high_performers = predictions >= threshold
        recommendations = []
        
        for i, (idx, row) in enumerate(self.X_test.iterrows()):
            actual_score = self.y_test.iloc[i]
            predicted_score = predictions[i]
            
            recommendation = {
                'node_id': self.data.iloc[idx]['node_id'],
                'actual_score': actual_score,
                'predicted_score': predicted_score,
                'recommendation': 'SELECT' if high_performers[i] else 'AVOID',
                'confidence': abs(predicted_score - threshold) / threshold,
                'reasons': self.explain_recommendation(row, predicted_score, threshold)
            }
            recommendations.append(recommendation)
        
        # Convert to DataFrame and sort by confidence
        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df.sort_values('confidence', ascending=False)
        
        print("=== TOP PEER RECOMMENDATIONS ===")
        print(rec_df.head(10)[['node_id', 'recommendation', 'predicted_score', 'confidence']])
        
        return rec_df
    
    def explain_recommendation(self, features, predicted_score, threshold):
        """Generate human-readable explanation for recommendation"""
        reasons = []
        
        # Energy efficiency
        if features['efficiency_ratio'] > 0.2:
            reasons.append(f"High efficiency ratio ({features['efficiency_ratio']:.3f})")
        elif features['efficiency_ratio'] < 0.1:
            reasons.append(f"Low efficiency ratio ({features['efficiency_ratio']:.3f})")
        
        # Upload speed
        if features['upload_speed'] > 18:
            reasons.append(f"Excellent upload speed ({features['upload_speed']:.1f} MB/s)")
        elif features['upload_speed'] < 8:
            reasons.append(f"Poor upload speed ({features['upload_speed']:.1f} MB/s)")
        
        # Energy consumption
        if features['energy_consumption'] < 50:
            reasons.append(f"Low energy consumption ({features['energy_consumption']:.1f}W)")
        elif features['energy_consumption'] > 90:
            reasons.append(f"High energy consumption ({features['energy_consumption']:.1f}W)")
        
        # Consistency
        if features['consistency'] > 0.8:
            reasons.append("Highly consistent performance")
        elif features['consistency'] < 0.5:
            reasons.append("Inconsistent performance")
        
        return "; ".join(reasons) if reasons else "Standard performance metrics"
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        report = {
            'statistical_tests': self.perform_statistical_tests(),
            'model_validation': self.validate_model_assumptions(),
            'performance_analysis': self.analyze_performance_distribution()
        }
        
        # Save report to file
        with open('../results/validation_report.txt', 'w') as f:
            f.write("=== EXPLAINABLE AI VALIDATION REPORT ===\n\n")
            
            f.write("1. STATISTICAL TESTS\n")
            for test_name, result in report['statistical_tests'].items():
                f.write(f"{test_name}: {result}\n")
            
            f.write("\n2. MODEL VALIDATION\n")
            for validation, result in report['model_validation'].items():
                f.write(f"{validation}: {result}\n")
            
            f.write("\n3. PERFORMANCE ANALYSIS\n")
            for metric, value in report['performance_analysis'].items():
                f.write(f"{metric}: {value}\n")
        
        print("Validation report saved to ../results/validation_report.txt")
        return report
    
    def perform_statistical_tests(self):
        """Perform statistical tests for validation"""
        tests = {}
        
        # Normality test for scores
        stat, p_value = stats.shapiro(self.data['score'].sample(min(5000, len(self.data))))
        tests['Score Normality (Shapiro-Wilk)'] = f"p-value: {p_value:.6f}, Normal: {p_value > 0.05}"
        
        # Correlation test between energy and speed
        corr, p_val = stats.pearsonr(self.data['energy_consumption'], self.data['upload_speed'])
        tests['Energy-Speed Correlation'] = f"r: {corr:.4f}, p-value: {p_val:.6f}"
        
        # Test for equal variances across performance categories
        groups = [group['score'].values for name, group in 
                 self.data.groupby('speed_category') if len(group) > 10]
        if len(groups) >= 2:
            stat, p_val = stats.levene(*groups)
            tests['Variance Homogeneity'] = f"p-value: {p_val:.6f}, Homogeneous: {p_val > 0.05}"
        
        return tests
    
    def validate_model_assumptions(self):
        """Validate machine learning model assumptions"""
        validations = {}
        
        if 'linear' in self.models:
            # Residual analysis for linear regression
            predictions = self.models['linear'].predict(self.X_test)
            residuals = self.y_test - predictions
            
            # Test residual normality
            stat, p_val = stats.shapiro(residuals)
            validations['Residual Normality'] = f"p-value: {p_val:.6f}"
            
            # Test homoscedasticity (Breusch-Pagan test approximation)
            residual_variance = np.var(residuals)
            validations['Residual Variance'] = f"{residual_variance:.6f}"
        
        # Cross-validation stability
        if 'random_forest' in self.models:
            cv_scores = cross_val_score(self.models['random_forest'], 
                                      self.X_train, self.y_train, cv=10)
            validations['CV Score Stability'] = f"Mean: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}"
        
        return validations
    
    def analyze_performance_distribution(self):
        """Analyze the distribution of performance metrics"""
        analysis = {}
        
        # Score distribution
        analysis['Score Mean'] = f"{self.data['score'].mean():.6f}"
        analysis['Score Std'] = f"{self.data['score'].std():.6f}"
        analysis['Score Skewness'] = f"{stats.skew(self.data['score']):.4f}"
        analysis['Score Kurtosis'] = f"{stats.kurtosis(self.data['score']):.4f}"
        
        # Performance categories
        perf_dist = self.data['speed_category'].value_counts(normalize=True)
        analysis['Speed Distribution'] = dict(perf_dist)
        
        return analysis

def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = ExplainableEnergyAnalyzer()
    
    # Ensure data is generated if not loaded
    if analyzer.data is None:
        analyzer.generate_synthetic_data()
    
    # Train models
    print("Training interpretable models...")
    model_results = analyzer.train_interpretable_models()
    
    # Generate explanations
    print("\n" + "="*60)
    print("EXPLAINABLE AI ANALYSIS")
    print("="*60)
    
    # Decision tree explanation
    analyzer.explain_decision_tree()
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    importance_df = analyzer.analyze_feature_importance()
    
    # SHAP analysis
    print("\nPerforming SHAP analysis...")
    try:
        shap_values = analyzer.shap_analysis()
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
    
    # Generate recommendations
    print("\nGenerating peer recommendations...")
    recommendations = analyzer.generate_peer_recommendations()
    
    # Validation report
    print("\nGenerating validation report...")
    validation_report = analyzer.generate_validation_report()
    
    print("\nExplainable AI analysis complete!")
    print("Check ../images/ for visualizations and ../results/ for detailed reports.")

if __name__ == "__main__":
    main()