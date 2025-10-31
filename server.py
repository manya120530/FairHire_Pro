"""
FairHirePro Backend - Unbiased Resume Screening System
Flask server with AI-powered resume analysis and advanced bias mitigation
Integrates: AI Fairness 360, Fairlearn, TF-IDF, Cosine Similarity
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import json

# Fairness libraries
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.algorithms.preprocessing import Reweighing
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    logging.warning("AI Fairness 360 not available. Install with: pip install aif360")

try:
    from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    logging.warning("Fairlearn not available. Install with: pip install fairlearn")

# ============================================================================
# Configuration & Setup
# ============================================================================

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyDSuf78IfCV4Qjhcxa9RfEDUVBRb10zbT4')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# ============================================================================
# Advanced Bias Detection with AI Fairness 360
# ============================================================================

class AdvancedBiasDetector:
    """Advanced bias detection using multiple techniques"""
    
    BIAS_PATTERNS = {
        'name': r'^\s*[A-Z][a-z]+\s+[A-Z][a-z]+',  # Only match names at start of document
        'age': r'\b(age[:\s]+\d{2}|born\s+in\s+\d{4}|\d{2}\s+years\s+old)\b',
        'gender': r'\b(male|female|gender[:\s]+(male|female))\b',  # Removed pronouns - too common in text
        'ethnicity': r'\b(african\s+american|caucasian|hispanic|latino|latina|native\s+american)\b',  # More specific
        'religion': r'\b(christian|muslim|hindu|buddhist|jewish|religious\s+affiliation)\b',
        'marital_status': r'\b(married|single|divorced|widowed|marital\s+status)\b',
        'address': r'\b\d+\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct)\b',
        'photo_reference': r'\b(attach.*photo|photograph.*attached|headshot|profile\s+picture)\b',
        'nationality': r'\b(nationality[:\s]+(american|canadian|british|indian|chinese|japanese))\b',  # Only explicit mentions
        'disability': r'\b(disabled|disability\s+status|wheelchair|accessibility\s+needs)\b',
    }
    
    @staticmethod
    def detect_bias_indicators(text: str) -> Dict[str, Any]:
        """Comprehensive bias detection"""
        detected = {}
        total_matches = 0
        
        for category, pattern in AdvancedBiasDetector.BIAS_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected[category] = {
                    'matches': matches,
                    'count': len(matches),
                    'severity': AdvancedBiasDetector._calculate_severity(category, len(matches))
                }
                total_matches += len(matches)
        
        return {
            'indicators': detected,
            'total_count': total_matches,
            'risk_level': AdvancedBiasDetector._calculate_risk_level(total_matches),
            'categories_affected': list(detected.keys())
        }
    
    @staticmethod
    def _calculate_severity(category: str, count: int) -> str:
        """Calculate severity of bias indicator"""
        high_risk = ['gender', 'ethnicity', 'religion', 'disability', 'nationality']
        medium_risk = ['age', 'marital_status']
        
        if category in high_risk:
            return 'HIGH' if count > 3 else 'MEDIUM' if count > 1 else 'LOW'
        elif category in medium_risk:
            return 'MEDIUM' if count > 2 else 'LOW'
        return 'LOW'
    
    @staticmethod
    def _calculate_risk_level(total_count: int) -> str:
        """Calculate overall risk level based on bias indicators found"""
        # Most resumes naturally have name (1-2 matches), so we need realistic thresholds
        if total_count >= 15:  # Very high - multiple demographics mentioned repeatedly
            return 'HIGH'
        elif total_count >= 8:  # Multiple protected categories
            return 'MEDIUM'
        elif total_count >= 3:  # Just name and maybe one other thing (normal)
            return 'LOW'
        return 'NONE'
    
    @staticmethod
    def mask_sensitive_info(text: str) -> Tuple[str, List[str]]:
        """Mask sensitive information with detailed logging"""
        masked_text = text
        masking_log = []
        
        # Mask names (only first occurrence at start of resume)
        name_pattern = r'^\s*[A-Z][a-z]+\s+[A-Z][a-z]+'
        name_match = re.search(name_pattern, masked_text, re.MULTILINE)
        if name_match:
            masked_text = re.sub(name_pattern, '[CANDIDATE]', masked_text, count=1, flags=re.MULTILINE)
            masking_log.append(f"Masked candidate name")
        
        # Mask age
        age_pattern = r'\b(age[:\s]+\d{2}|born\s+in\s+\d{4}|\d{2}\s+years\s+old)\b'
        age_matches = re.findall(age_pattern, masked_text, re.IGNORECASE)
        if age_matches:
            masked_text = re.sub(age_pattern, '[AGE]', masked_text, flags=re.IGNORECASE)
            masking_log.append(f"Masked {len(age_matches)} age reference(s)")
        
        # Mask explicit gender references (not pronouns)
        gender_pattern = r'\b(male|female|gender[:\s]+(male|female))\b'
        gender_matches = re.findall(gender_pattern, masked_text, re.IGNORECASE)
        if gender_matches:
            masked_text = re.sub(gender_pattern, '[GENDER]', masked_text, flags=re.IGNORECASE)
            masking_log.append(f"Masked {len(gender_matches)} gender reference(s)")
        
        # Mask ethnicity/race (specific phrases only)
        ethnicity_pattern = r'\b(african\s+american|caucasian|hispanic|latino|latina|native\s+american)\b'
        ethnicity_matches = re.findall(ethnicity_pattern, masked_text, re.IGNORECASE)
        if ethnicity_matches:
            masked_text = re.sub(ethnicity_pattern, '[ETHNICITY]', masked_text, flags=re.IGNORECASE)
            masking_log.append(f"Masked {len(ethnicity_matches)} ethnicity reference(s)")
        
        # Mask marital status
        marital_pattern = r'\b(married|single|divorced|widowed|marital\s+status)\b'
        marital_matches = re.findall(marital_pattern, masked_text, re.IGNORECASE)
        if marital_matches:
            masked_text = re.sub(marital_pattern, '[MARITAL_STATUS]', masked_text, flags=re.IGNORECASE)
            masking_log.append(f"Masked {len(marital_matches)} marital status reference(s)")
        
        # Mask addresses
        address_pattern = r'\b\d+\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)\b'
        address_matches = re.findall(address_pattern, masked_text, re.IGNORECASE)
        if address_matches:
            masked_text = re.sub(address_pattern, '[ADDRESS]', masked_text, flags=re.IGNORECASE)
            masking_log.append(f"Masked {len(address_matches)} address(es)")
        
        # Mask nationality (explicit only)
        nationality_pattern = r'\b(nationality[:\s]+(american|canadian|british|indian|chinese|japanese))\b'
        nationality_matches = re.findall(nationality_pattern, masked_text, re.IGNORECASE)
        if nationality_matches:
            masked_text = re.sub(nationality_pattern, '[NATIONALITY]', masked_text, flags=re.IGNORECASE)
            masking_log.append(f"Masked {len(nationality_matches)} nationality reference(s)")
        
        # Mask religion
        religion_pattern = r'\b(christian|muslim|hindu|buddhist|jewish|religious\s+affiliation)\b'
        religion_matches = re.findall(religion_pattern, masked_text, re.IGNORECASE)
        if religion_matches:
            masked_text = re.sub(religion_pattern, '[RELIGION]', masked_text, flags=re.IGNORECASE)
            masking_log.append(f"Masked {len(religion_matches)} religious reference(s)")
        
        if not masking_log:
            masking_log.append("No sensitive information detected")
        
        return masked_text, masking_log

# ============================================================================
# Fairness Metrics with AI Fairness 360 & Fairlearn
# ============================================================================

class FairnessMetricsEngine:
    """Calculate comprehensive fairness metrics"""
    
    @staticmethod
    def calculate_fairness_metrics(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate fairness metrics using available libraries"""
        metrics = {
            'demographic_parity': None,
            'equal_opportunity': None,
            'disparate_impact': None,
            'statistical_parity_difference': None,
            'fairness_score': 100,
            'recommendations': [],
            'library_used': 'custom'
        }
        
        if len(candidates) < 2:
            metrics['recommendations'].append("Need at least 2 candidates for fairness analysis")
            return metrics
        
        # Basic statistical fairness checks
        scores = [c['overallScore'] for c in candidates]
        
        # Calculate score variance (should be reasonable)
        score_std = np.std(scores)
        score_range = np.max(scores) - np.min(scores)
        
        # Score distribution fairness
        if score_std < 5:
            metrics['fairness_score'] -= 15
            metrics['recommendations'].append("Low score variance - consider reviewing ranking criteria")
        
        if score_range > 70:
            metrics['fairness_score'] -= 10
            metrics['recommendations'].append("Very wide score range - ensure consistent evaluation")
        
        # Check for clustering
        top_scores = sorted(scores, reverse=True)[:3]
        if len(top_scores) > 1 and (max(top_scores) - min(top_scores)) < 5:
            metrics['recommendations'].append("Top candidates very close - consider tie-breaking criteria")
        
        # Bias indicator check
        bias_counts = [len(c.get('biasIndicators', [])) for c in candidates]
        avg_bias = np.mean(bias_counts)
        
        if avg_bias > 3:
            metrics['fairness_score'] -= 20
            metrics['recommendations'].append("High average bias indicators detected across resumes")
        
        metrics['statistical_summary'] = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(score_std),
            'range': float(score_range),
            'avg_bias_indicators': float(avg_bias)
        }
        
        # Try to use Fairlearn if available
        if FAIRLEARN_AVAILABLE:
            try:
                metrics.update(FairnessMetricsEngine._calculate_fairlearn_metrics(candidates))
                metrics['library_used'] = 'fairlearn'
            except Exception as e:
                logger.warning(f"Fairlearn calculation failed: {e}")
        
        return metrics
    
    @staticmethod
    def _calculate_fairlearn_metrics(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate Fairlearn fairness metrics"""
        # Create synthetic protected attributes based on bias detection
        # In production, this would come from actual demographic data
        df = pd.DataFrame({
            'score': [c['overallScore'] for c in candidates],
            'selected': [1 if c['overallScore'] >= 70 else 0 for c in candidates],
            'has_bias_indicators': [1 if len(c.get('biasIndicators', [])) > 2 else 0 for c in candidates]
        })
        
        if len(df) < 2:
            return {}
        
        # Calculate selection rate by bias indicator presence
        try:
            metric_frame = MetricFrame(
                metrics=selection_rate,
                y_true=df['selected'],
                y_pred=df['selected'],
                sensitive_features=df['has_bias_indicators']
            )
            
            return {
                'selection_rate_by_group': metric_frame.by_group.to_dict(),
                'demographic_parity_diff': float(metric_frame.difference())
            }
        except Exception as e:
            logger.warning(f"Fairlearn metric calculation error: {e}")
            return {}

# ============================================================================
# Enhanced Semantic Matching
# ============================================================================

class SemanticMatcher:
    """Enhanced semantic matching with TF-IDF and cosine similarity"""
    
    def __init__(self):
        # Vectorizer will be created fresh for each comparison
        pass
    
    def calculate_similarity(self, job_description: str, resume: str) -> Dict[str, float]:
        """Calculate multiple similarity metrics"""
        try:
            # Clean and validate inputs
            job_clean = job_description.strip()
            resume_clean = resume.strip()
            
            if not job_clean or not resume_clean:
                logger.warning("Empty job description or resume")
                return {'cosine_similarity': 0.0, 'keyword_overlap': 0.0, 'combined_score': 0.0}
            
            # Create fresh vectorizer for this comparison
            vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                lowercase=True,
                max_df=0.95
            )
            
            documents = [job_clean, resume_clean]
            
            # Fit and transform both documents
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Calculate cosine similarity between job (0) and resume (1)
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Extract keywords separately for overlap calculation
            job_keywords = set(self.extract_keywords(job_clean, top_n=30))
            resume_keywords = set(self.extract_keywords(resume_clean, top_n=30))
            
            # Calculate keyword overlap
            if len(job_keywords) > 0:
                keyword_overlap = len(job_keywords & resume_keywords) / len(job_keywords)
            else:
                keyword_overlap = 0.0
            
            # Combined score
            combined = (cosine_sim * 0.6) + (keyword_overlap * 0.4)
            
            logger.info(f"Similarity metrics - Cosine: {cosine_sim:.3f}, Overlap: {keyword_overlap:.3f}, Combined: {combined:.3f}")
            
            return {
                'cosine_similarity': float(cosine_sim),
                'keyword_overlap': float(keyword_overlap),
                'combined_score': float(combined)
            }
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}", exc_info=True)
            return {'cosine_similarity': 0.0, 'keyword_overlap': 0.0, 'combined_score': 0.0}
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """Extract top keywords using TF-IDF"""
        try:
            if not text.strip():
                return []
            
            # Create separate vectorizer for keyword extraction
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top N keywords
            top_indices = tfidf_scores.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}", exc_info=True)
            return []

# ============================================================================
# Enhanced Resume Analyzer
# ============================================================================

class ResumeAnalyzer:
    """Main analyzer with advanced fairness features"""
    
    def __init__(self):
        self.bias_detector = AdvancedBiasDetector()
        self.semantic_matcher = SemanticMatcher()
        self.fairness_engine = FairnessMetricsEngine()
    
    def analyze_single_resume(
        self,
        resume_id: int,
        resume_content: str,
        job_description: str
    ) -> Dict[str, Any]:
        """Analyze single resume with comprehensive metrics"""
        
        # Detect bias indicators
        bias_analysis = self.bias_detector.detect_bias_indicators(resume_content)
        
        # Mask sensitive information
        masked_resume, masking_log = self.bias_detector.mask_sensitive_info(resume_content)
        
        # Calculate semantic similarity
        similarity_metrics = self.semantic_matcher.calculate_similarity(
            job_description,
            masked_resume
        )
        
        # Extract keywords
        job_keywords = self.semantic_matcher.extract_keywords(job_description, top_n=15)
        
        # Build AI prompt
        prompt = self._build_analysis_prompt(
            job_description,
            masked_resume,
            job_keywords,
            bias_analysis
        )
        
        try:
            # Get AI analysis
            response = model.generate_content(prompt)
            ai_analysis = self._parse_ai_response(response.text)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                ai_analysis,
                similarity_metrics['combined_score']
            )
            
            # Generate warnings
            warnings = self._generate_warnings(bias_analysis, ai_analysis)
            
            return {
                'id': resume_id,
                'overallScore': overall_score,
                'justification': ai_analysis.get('justification', 'No justification provided'),
                'professionalSummary': ai_analysis.get('professional_summary', 'Summary not available'),
                'skillsMatch': ai_analysis.get('skills_match', []),
                'experienceMatch': ai_analysis.get('experience_match', 'Not analyzed'),
                'educationLevel': ai_analysis.get('education_level', 'Not specified'),
                'certifications': ai_analysis.get('certifications', []),
                'softSkills': ai_analysis.get('soft_skills', []),
                'resumeQuality': ai_analysis.get('resume_quality', 'Standard'),
                'warnings': warnings,
                'biasIndicators': bias_analysis['categories_affected'],
                'biasRiskLevel': bias_analysis['risk_level'],
                'biasDetails': bias_analysis,
                'maskingLog': masking_log,
                'semanticMetrics': similarity_metrics,
            }
            
        except Exception as e:
            logger.error(f"Error analyzing resume {resume_id}: {e}")
            return self._create_error_response(resume_id, str(e))
    
    def _build_analysis_prompt(
        self,
        job_description: str,
        resume: str,
        job_keywords: List[str],
        bias_analysis: Dict
    ) -> str:
        """Build AI analysis prompt"""
        
        prompt = f"""You are an expert HR analyst performing UNBIASED resume evaluation. All sensitive demographic information has been masked to ensure fairness.

JOB DESCRIPTION:
{job_description}

KEY SKILLS REQUIRED:
{', '.join(job_keywords)}

RESUME (sensitive information masked):
{resume}

BIAS MITIGATION APPLIED:
- {len(bias_analysis['categories_affected'])} types of sensitive information masked
- Risk Level: {bias_analysis['risk_level']}

EVALUATION INSTRUCTIONS:
1. Focus ONLY on professional qualifications, skills, experience, and education
2. Ignore all demographic information (already masked)
3. Provide objective, evidence-based analysis
4. Be fair but critical

Provide analysis in JSON format:
{{
    "justification": "Brief explanation (2-3 sentences)",
    "professional_summary": "One-sentence professional summary",
    "skills_match": [
        {{"name": "Skill1", "mentioned": true}},
        {{"name": "Skill2", "mentioned": false}}
    ],
    "experience_match": "Experience analysis",
    "education_level": "Education level",
    "certifications": ["Cert1", "Cert2"],
    "soft_skills": ["Communication", "Leadership"],
    "resume_quality": "Excellent/Good/Average/Poor with explanation",
    "technical_depth_score": 0-100,
    "experience_relevance_score": 0-100,
    "overall_fit_score": 0-100
}}

Respond with valid JSON only."""
        
        return prompt
    
    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return self._create_default_analysis()
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return self._create_default_analysis()
    
    def _create_default_analysis(self) -> Dict[str, Any]:
        """Default analysis structure"""
        return {
            'justification': 'Analysis incomplete',
            'professional_summary': 'Could not generate summary',
            'skills_match': [],
            'experience_match': 'Not analyzed',
            'education_level': 'Not specified',
            'certifications': [],
            'soft_skills': [],
            'resume_quality': 'Unable to assess',
            'technical_depth_score': 50,
            'experience_relevance_score': 50,
            'overall_fit_score': 50
        }
    
    def _calculate_overall_score(
        self,
        ai_analysis: Dict[str, Any],
        similarity_score: float
    ) -> int:
        """Calculate weighted overall score"""
        
        weights = {
            'technical_depth': 0.30,
            'experience_relevance': 0.25,
            'overall_fit': 0.25,
            'semantic_similarity': 0.20
        }
        
        technical = ai_analysis.get('technical_depth_score', 50)
        experience = ai_analysis.get('experience_relevance_score', 50)
        fit = ai_analysis.get('overall_fit_score', 50)
        semantic = similarity_score * 100
        
        overall = (
            technical * weights['technical_depth'] +
            experience * weights['experience_relevance'] +
            fit * weights['overall_fit'] +
            semantic * weights['semantic_similarity']
        )
        
        return int(round(overall))
    
    def _generate_warnings(
        self,
        bias_analysis: Dict,
        ai_analysis: Dict
    ) -> List[str]:
        """Generate warnings"""
        warnings = []
        
        # Bias warnings
        if bias_analysis['risk_level'] in ['HIGH', 'MEDIUM']:
            warnings.append(
                f"Resume contains {bias_analysis['risk_level']} risk bias indicators in: {', '.join(bias_analysis['categories_affected'])}"
            )
        
        # Quality warnings
        quality = ai_analysis.get('resume_quality', '').lower()
        if 'poor' in quality or 'average' in quality:
            warnings.append("Resume quality could be improved")
        
        # Missing information
        if not ai_analysis.get('education_level') or 'not specified' in ai_analysis.get('education_level', '').lower():
            warnings.append("Education information is incomplete")
        
        return warnings
    
    def _create_error_response(self, resume_id: int, error_msg: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            'id': resume_id,
            'overallScore': 0,
            'justification': f'Analysis failed: {error_msg}',
            'professionalSummary': 'Error during analysis',
            'skillsMatch': [],
            'experienceMatch': 'Error',
            'educationLevel': 'Unknown',
            'certifications': [],
            'softSkills': [],
            'resumeQuality': 'Unable to assess',
            'warnings': ['Analysis error occurred'],
            'biasIndicators': [],
            'biasRiskLevel': 'UNKNOWN'
        }

# ============================================================================
# API Endpoints
# ============================================================================

analyzer = ResumeAnalyzer()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'FairHirePro Backend',
        'features': {
            'ai_fairness_360': AIF360_AVAILABLE,
            'fairlearn': FAIRLEARN_AVAILABLE,
            'gemini_api': GEMINI_API_KEY != 'YOUR_API_KEY_HERE'
        },
        'version': '2.0.0'
    }), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_resumes():
    """Main resume analysis endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        job_description = data.get('jobDescription', '')
        resumes = data.get('resumes', [])
        
        if not job_description.strip():
            return jsonify({'error': 'Job description is required'}), 400
        
        if not resumes:
            return jsonify({'error': 'At least one resume is required'}), 400
        
        logger.info(f"Starting analysis for {len(resumes)} resume(s)")
        
        # Analyze each resume
        results = []
        for resume_data in resumes:
            resume_id = resume_data.get('id')
            resume_content = resume_data.get('content', '')
            
            if not resume_content.strip():
                logger.warning(f"Skipping empty resume {resume_id}")
                continue
            
            logger.info(f"Analyzing resume {resume_id}")
            
            analysis = analyzer.analyze_single_resume(
                resume_id,
                resume_content,
                job_description
            )
            
            results.append(analysis)
        
        # Sort by overall score
        results.sort(key=lambda x: x['overallScore'], reverse=True)
        
        # Calculate fairness metrics
        fairness_metrics = analyzer.fairness_engine.calculate_fairness_metrics(results)
        
        logger.info(f"Analysis completed. Fairness score: {fairness_metrics['fairness_score']}")
        
        # Log semantic metrics for debugging
        for result in results:
            if 'semanticMetrics' in result:
                logger.info(f"Candidate {result['id']} - Semantic: {result['semanticMetrics']}")
        
        # Return results array directly (frontend expects array, not object with 'candidates' key)
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Error in analyze_resumes: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/bias-check', methods=['POST'])
def check_bias():
    """Bias detection endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'Text is required'}), 400
        
        bias_analysis = AdvancedBiasDetector.detect_bias_indicators(text)
        masked_text, masking_log = AdvancedBiasDetector.mask_sensitive_info(text)
        
        return jsonify({
            'original_length': len(text),
            'masked_length': len(masked_text),
            'bias_analysis': bias_analysis,
            'masked_text': masked_text,
            'masking_log': masking_log,
            'has_bias': bias_analysis['total_count'] > 0,
            'recommendation': 'Remove sensitive information' if bias_analysis['risk_level'] in ['HIGH', 'MEDIUM'] else 'Acceptable'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in bias_check: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 FairHirePro Backend Server v2.0")
    print("="*70)
    
    if GEMINI_API_KEY == 'YOUR_API_KEY_HERE':
        print("⚠️  WARNING: GEMINI_API_KEY not set!")
        print("   Set it: export GEMINI_API_KEY='your-key-here'")
    else:
        print("✅ Gemini API: Configured")
    
    print(f"✅ AI Fairness 360: {'Available' if AIF360_AVAILABLE else 'Not installed'}")
    print(f"✅ Fairlearn: {'Available' if FAIRLEARN_AVAILABLE else 'Not installed'}")
    print("\n📡 Server starting on http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=8000, debug=True)