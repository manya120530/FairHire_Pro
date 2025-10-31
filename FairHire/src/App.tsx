import React, { useState, useCallback,  useMemo, useEffect } from 'react';
import * as d from 'pdfjs-dist';
import { ResponsiveContainer, Legend, Tooltip, PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';


// ============================================================================
// Global Declarations & Setup
// ============================================================================

declare global {
  interface Window {
    mammoth: any;
    pdfjsLib: typeof d;
  }
}

// The worker is shipped separately due to its size.
window.pdfjsLib = d;
window.pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@4.4.168/build/pdf.worker.min.mjs`;

// ============================================================================
// Type Definitions
// ============================================================================

interface Skill {
  name: string;
  mentioned: boolean;
}

interface SemanticMetrics {
  cosine_similarity: number;
  keyword_overlap: number;
  combined_score: number;
}

interface BiasDetails {
    indicators: Record<string, any>;
    total_count: number;
    risk_level: 'NONE' | 'LOW' | 'MEDIUM' | 'HIGH';
    categories_affected: string[];
}

interface Candidate {
  id: number;
  overallScore: number;
  justification: string;
  professionalSummary: string;
  skillsMatch: Skill[];
  experienceMatch: string;
  educationLevel: string;
  certifications: string[];
  softSkills: string[];
  resumeQuality: string;
  warnings: string[];
  biasIndicators: string[];
  biasRiskLevel: 'NONE' | 'LOW' | 'MEDIUM' | 'HIGH' | 'UNKNOWN';
  biasDetails: BiasDetails;
  maskingLog: string[];
  semanticMetrics: SemanticMetrics;
  technical_depth_score?: number;
  experience_relevance_score?: number;
  overall_fit_score?: number;
}

interface FairnessMetrics {
    demographic_parity: number | null;
    equal_opportunity: number | null;
    disparate_impact: number | null;
    statistical_parity_difference: number | null;
    fairness_score: number;
    recommendations: string[];
    library_used: string;
    statistical_summary: {
        mean_score: number;
        std_score: number;
        range: number;
        avg_bias_indicators: number;
    };
}

interface AnalysisResponse {
    candidates: Candidate[];
    fairnessMetrics: FairnessMetrics;
    metadata: Record<string, any>;
}

interface ResumeFile {
  file: File;
  name: string;
  status: 'pending' | 'parsing' | 'success' | 'error' | 'duplicate';
  content: string | null;
  hash: string | null;
  error?: string;
}

// ============================================================================
// Client-side Fairness Metrics Calculation
// ============================================================================

const calculateMean = (arr: number[]): number => arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length;

const calculateStdDev = (arr: number[]): number => {
  if (arr.length < 2) return 0;
  const mean = calculateMean(arr);
  const variance = arr.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / arr.length;
  return Math.sqrt(variance);
};

const calculateFrontendFairnessMetrics = (candidates: Candidate[]): FairnessMetrics => {
    const metrics: FairnessMetrics = {
        demographic_parity: null,
        equal_opportunity: null,
        disparate_impact: null,
        statistical_parity_difference: null,
        fairness_score: 100,
        recommendations: [],
        library_used: 'frontend-statistical',
        statistical_summary: {
            mean_score: 0,
            std_score: 0,
            range: 0,
            avg_bias_indicators: 0,
        },
    };

    if (candidates.length < 2) {
        metrics.recommendations.push("Need at least 2 candidates for fairness analysis");
        return metrics;
    }

    const scores = candidates.map(c => c.overallScore);
    const score_std = calculateStdDev(scores);
    const score_range = Math.max(...scores) - Math.min(...scores);

    if (score_std < 5) {
        metrics.fairness_score -= 15;
        metrics.recommendations.push("Low score variance - consider reviewing ranking criteria");
    }

    if (score_range > 70) {
        metrics.fairness_score -= 10;
        metrics.recommendations.push("Very wide score range - ensure consistent evaluation");
    }

    const top_scores = [...scores].sort((a, b) => b - a).slice(0, 3);
    if (top_scores.length > 1 && (Math.max(...top_scores) - Math.min(...top_scores)) < 5) {
        metrics.recommendations.push("Top candidates very close - consider tie-breaking criteria");
    }

    const bias_counts = candidates.map(c => c.biasIndicators?.length || 0);
    const avg_bias = calculateMean(bias_counts);

    if (avg_bias > 3) {
        metrics.fairness_score -= 20;
        metrics.recommendations.push("High average bias indicators detected across resumes");
    }

    metrics.statistical_summary = {
        mean_score: calculateMean(scores),
        std_score: score_std,
        range: score_range,
        avg_bias_indicators: avg_bias,
    };
    
    // Ensure score is not negative
    metrics.fairness_score = Math.max(0, metrics.fairness_score);

    return metrics;
}

// ============================================================================
// File Parsing Utilities
// ============================================================================

async function getFileHash(file: File): Promise<string> {
    const buffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

async function parsePdf(file: File): Promise<string> {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await window.pdfjsLib.getDocument(arrayBuffer).promise;
    let text = '';
    for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        text += content.items.map((item: any) => 'str' in item ? item.str : '').join(' ');
    }
    return text;
}

async function parseDocx(file: File): Promise<string> {
    if (!window.mammoth) {
        throw new Error("DOCX parsing library (mammoth.js) is not loaded.");
    }
    try {
        const arrayBuffer = await file.arrayBuffer();
        const result = await window.mammoth.extractRawText({ arrayBuffer });
        return result.value;
    } catch (e) {
        console.error("Mammoth.js parsing error:", e);
        throw new Error("Failed to parse .docx file. It may be corrupt or an unsupported format.");
    }
}

function parseTxt(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            if (event.target && typeof event.target.result === 'string') {
                resolve(event.target.result);
            } else {
                reject(new Error("Failed to read file content as text."));
            }
        };
        reader.onerror = (error) => {
            reject(new Error("Error reading file: " + error));
        };
        reader.readAsText(file);
    });
}

const parseResumeFile = async (file: File): Promise<{ content: string, hash: string }> => {
    if (file.size === 0) {
        throw new Error("File is empty.");
    }

    let content: string;
    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    
    switch (fileExtension) {
        case 'pdf':
            content = await parsePdf(file);
            break;
        case 'docx':
        case 'doc':
            content = await parseDocx(file);
            break;
        case 'txt':
            content = await parseTxt(file);
            break;
        default:
            throw new Error(`Unsupported file type: '.${fileExtension}'. Please upload .pdf, .docx, or .txt files.`);
    }

    const hash = await getFileHash(file);
    return { content, hash };
};

// ============================================================================
// API Service
// ============================================================================

const analyzeResumes = async (jobDescription: string, resumeFiles: ResumeFile[]): Promise<Candidate[]> => {
  const payload = {
      jobDescription,
      resumes: resumeFiles
        .filter(r => r.status === 'success')
        .map((resume, index) => ({
          id: index + 1,
          content: resume.content!
      }))
  };

  const response = await fetch('http://localhost:5000/api/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorMessage = `Analysis failed: Server responded with status ${response.status}`;
    try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorData.details || errorMessage;
    } catch (e) {
        // Could not parse error JSON
    }
    throw new Error(errorMessage);
  }

  const results: Candidate[] = await response.json();
  return results;
};


// ============================================================================
// Icon Components
// ============================================================================
const BriefcaseIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 14.15v4.07a2.25 2.25 0 01-2.25 2.25H5.92a2.25 2.25 0 01-2.25-2.25v-4.07m16.5 0a2.25 2.25 0 00-2.25-2.25H5.92a2.25 2.25 0 00-2.25 2.25m16.5 0v-2.07a2.25 2.25 0 00-2.25-2.25H5.92a2.25 2.25 0 00-2.25 2.25v2.07m16.5 0v-2.83a2.25 2.25 0 00-2.25-2.25H5.92a2.25 2.25 0 00-2.25 2.25v2.83m16.5 0h-16.5m16.5 0a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H5.92a2.25 2.25 0 00-2.25 2.25v4.67a2.25 2.25 0 002.25 2.25h12.36z" />
    </svg>
);
const InformationCircleIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
);
const ArrowPathIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0011.667 0l3.181-3.183m-4.991-2.696L7.985 5.987m0 0A8.25 8.25 0 0112 3.75a8.25 8.25 0 014.015 2.237l-1.55 1.55m-5.465 5.465a3 3 0 000 4.243m2.122-2.122a3 3 0 00-4.242 0" />
    </svg>
);
const CheckCircleIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
);
const XCircleIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
);
const StarIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" {...props}>
        <path fillRule="evenodd" d="M10.788 3.21c.448-1.077 1.976-1.077 2.424 0l2.082 5.007 5.404.433c1.164.093 1.636 1.545.749 2.305l-4.117 3.527 1.257 5.273c.271 1.136-.964 2.033-1.96 1.425L12 18.354 7.373 21.18c-.996.608-2.231-.29-1.96-1.425l1.257-5.273-4.117-3.527c-.887-.76-.415-2.212.749-2.305l5.404-.433 2.082-5.007z" clipRule="evenodd" />
    </svg>
);
const LightBulbIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.375 6.375 0 00-6.375-6.375M12 12.75a6.375 6.375 0 016.375-6.375M12 12.75v-5.25m0 5.25a6.375 6.375 0 00-6.375 6.375m6.375-6.375a6.375 6.375 0 016.375 6.375m-6.375-6.375v-5.25m0 5.25h.008v.008H12v-.008z" />
    </svg>
);
const ChevronDownIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
    </svg>
);
const AcademicCapIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path d="M12 14.25c-3.728 0-7.152-1.576-9.75-4.173A12.75 12.75 0 0112 3.75c4.969 0 9.242 2.82 11.25 6.883-2.58 2.534-6.004 4.117-9.75 4.117z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 14.25a8.962 8.962 0 01-1.295-.122M12 14.25c.43 0 .854-.04 1.295-.122m-1.295.122v5.625c0 1.24.982 2.25 2.18 2.25H15a2.25 2.25 0 002.25-2.25v-1.28c0-.62-.25-1.208-.67-1.633a8.953 8.953 0 00-2.08-1.556 8.953 8.953 0 00-2.08 1.556c-.42.425-.67.97-.67 1.633v1.28a2.25 2.25 0 002.25 2.25h.538a2.18 2.18 0 002.18-2.25v-5.625" />
    </svg>
);
const ShieldCheckIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.286zm0 13.036h.008v.008h-.008v-.008z" />
    </svg>
);
const ExclamationTriangleIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
    </svg>
);
const XMarkIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
);
const ArrowUpTrayIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
  </svg>
);
const DocumentTextIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m9.15 12.75h-5.625M9 7.5h3.75M9 10.5h3.75m-3.75 3h3.75M5.625 21V3.375c0-.621.504-1.125 1.125-1.125H16.5a1.125 1.125 0 011.125 1.125v17.25c0 .621-.504 1.125-1.125 1.125H6.75A1.125 1.125 0 015.625 21z" />
  </svg>
);
const SpinnerIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" {...props}>
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);
const ExclamationCircleIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
    </svg>
);
const DocumentDuplicateIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 01-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 011.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 00-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 01-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 00-3.375-3.375h-1.5a1.125 1.125 0 01-1.125-1.125v-1.5a3.375 3.375 0 00-3.375-3.375H9.75" />
    </svg>
);
const SparklesIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.898 20.562L16.25 21.75l-.648-1.188a2.25 2.25 0 01-1.47-1.47l-1.188-.648 1.188-.648a2.25 2.25 0 011.47-1.47l.648-1.188.648 1.188a2.25 2.25 0 011.47 1.47l1.188.648-1.188.648a2.25 2.25 0 01-1.47 1.47z" />
    </svg>
);
const LinkIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M13.19 8.688a4.5 4.5 0 011.242 7.244l-4.5 4.5a4.5 4.5 0 01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5 4.5 0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244" />
    </svg>
);

// ============================================================================
// UI Components
// ============================================================================

const BrandPanel: React.FC<{ onInfoClick: () => void }> = ({ onInfoClick }) => (
    <div className="hidden lg:flex w-1/3 bg-indigo-600 text-white p-12 flex-col justify-between rounded-l-2xl">
        <div>
            <div className="flex items-center gap-3">
                <BriefcaseIcon className="h-8 w-8" />
                <h1 className="text-3xl font-bold tracking-tight">FairHirePro</h1>
            </div>
            <div className="mt-12 space-y-4">
                <h2 className="text-5xl font-extrabold leading-tight">Fast.</h2>
                <h2 className="text-5xl font-extrabold leading-tight">Simple.</h2>
                <h2 className="text-5xl font-extrabold leading-tight">Beautiful.</h2>
            </div>
            <p className="mt-8 text-indigo-200">Leveraging NLP and Artificial Intelligence to bring fairness and efficiency to your hiring process.</p>
        </div>
        <div className="text-sm">
            <button onClick={onInfoClick} className="flex items-center gap-2 text-indigo-200 hover:text-white transition-colors">
                <InformationCircleIcon className="w-5 h-5" />
                About FairHirePro
            </button>
        </div>
    </div>
);

const JobDescriptionInput: React.FC<{ jobDescription: string; setJobDescription: (v: string) => void; isProcessing: boolean; }> = ({ jobDescription, setJobDescription, isProcessing }) => {
    const templates: Record<string, string> = {
        'swe': `**Job Title:** Senior Software Engineer (Frontend)\n\n**Location:** Remote\n\n**About Us:** We are a fast-growing tech company revolutionizing the [Industry Name] space. We value collaboration, innovation, and a user-first mindset.\n\n**Responsibilities:**\n- Develop and maintain user-facing features using React and TypeScript.\n- Collaborate with product managers, designers, and backend engineers to create a seamless user experience.\n- Write clean, efficient, and well-tested code.\n- Mentor junior engineers and contribute to our engineering culture.\n- Optimize applications for maximum speed and scalability.\n\n**Qualifications:**\n- 5+ years of professional experience in frontend development.\n- Deep expertise in React, TypeScript, and modern JavaScript (ES6+).\n- Strong understanding of HTML5, CSS3, and responsive design principles.\n- Experience with state management libraries like Redux or Zustand.\n- Familiarity with testing frameworks such as Jest and React Testing Library.\n- Excellent problem-solving and communication skills.`,
        'pm': `**Job Title:** Product Manager\n\n**Location:** New York, NY (Hybrid)\n\n**About Us:** We are a market leader in [Industry Name], dedicated to creating products that our customers love. We believe in data-driven decisions and agile development.\n\n**Responsibilities:**\n- Define product vision, strategy, and roadmap.\n- Gather and prioritize product and customer requirements.\n- Work closely with engineering, design, marketing, and sales teams to ensure successful product launches.\n- Analyze market trends and competitive landscape.\n- Define and analyze metrics that inform the success of products.\n\n**Qualifications:**\n- 3+ years of experience in product management, preferably in a SaaS environment.\n- Proven track record of managing all aspects of a successful product throughout its lifecycle.\n- Strong technical background with an understanding of software development principles.\n- Excellent written and verbal communication skills.\n- Experience with Agile/Scrum methodologies.`,
    };

    return (
        <div className="space-y-3">
            <div className="flex justify-between items-center">
                <div>
                    <label className="text-sm font-semibold text-gray-700">1. Job Description</label>
                    <p className="text-gray-500 text-sm">
                        Paste the job description below or use a template to get started.
                    </p>
                </div>
                <select 
                    onChange={(e) => setJobDescription(templates[e.target.value] || "")}
                    disabled={isProcessing}
                    className="bg-gray-50 border border-gray-300 rounded-md px-3 py-1.5 text-sm text-gray-700 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                    aria-label="Job description templates"
                >
                    <option value="">Template...</option>
                    <option value="swe">Software Engineer</option>
                    <option value="pm">Product Manager</option>
                </select>
            </div>
            <textarea
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                disabled={isProcessing}
                placeholder="e.g., Senior Frontend Engineer at a fast-growing tech startup..."
                className="w-full h-40 p-3 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none transition-shadow disabled:opacity-50 disabled:cursor-not-allowed text-gray-800"
                aria-label="Job Description Input"
            />
        </div>
    );
};

const ResumeInputSection: React.FC<{ onFilesAdded: (files: FileList) => void; isProcessing: boolean; }> = ({ onFilesAdded, isProcessing }) => {
    const [isDragging, setIsDragging] = useState(false);
    
    const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        event.stopPropagation();
        setIsDragging(false);
        if (event.dataTransfer.files && !isProcessing) {
            onFilesAdded(event.dataTransfer.files);
        }
    }, [onFilesAdded, isProcessing]);
    
    const handleDragEvents = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        event.stopPropagation();
        if (event.type === 'dragenter' || event.type === 'dragover') {
            if (!isProcessing) setIsDragging(true);
        } else if (event.type === 'dragleave') {
            setIsDragging(false);
        }
    };

    return (
        <div className="space-y-3">
            <div>
                 <label className="text-sm font-semibold text-gray-700">2. Upload Resumes</label>
                 <p className="text-gray-500 text-sm">
                    Drag and drop or click to upload candidate resumes.
                </p>
            </div>
            <div 
                onDrop={handleDrop} 
                onDragEnter={handleDragEvents}
                onDragOver={handleDragEvents}
                onDragLeave={handleDragEvents}
                className={`border-2 border-dashed rounded-lg p-6 text-center bg-gray-50 transition-colors ${isDragging ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300 hover:border-indigo-400'}`}
            >
                <label htmlFor="resume-upload" className={isProcessing ? 'cursor-not-allowed' : 'cursor-pointer'}>
                    <div className="flex flex-col items-center justify-center text-gray-500">
                        <ArrowUpTrayIcon className="w-10 h-10 mb-2 text-gray-400" />
                        <span className="font-semibold text-indigo-600">Click to upload</span>
                        <span>or drag and drop</span>
                        <span className="text-xs mt-1">(PDF, DOCX, TXT)</span>
                    </div>
                    <input
                        id="resume-upload"
                        type="file"
                        multiple
                        accept=".pdf,.doc,.docx,text/plain"
                        className="hidden"
                        onChange={(e) => {
                            if (e.target.files) onFilesAdded(e.target.files);
                            e.target.value = '';
                        }}
                        disabled={isProcessing}
                    />
                </label>
            </div>
        </div>
    );
};

const FileStatusItem: React.FC<{file: ResumeFile; onRemove: () => void; isProcessing: boolean;}> = ({ file, onRemove, isProcessing }) => {
    let statusIcon, statusColor, statusText;

    switch (file.status) {
        case 'parsing':
            statusIcon = <SpinnerIcon className="w-5 h-5 animate-spin text-indigo-500" />;
            statusColor = 'text-gray-500';
            statusText = 'Parsing...';
            break;
        case 'success':
            statusIcon = <CheckCircleIcon className="w-5 h-5 text-green-500" />;
            statusColor = 'text-green-600';
            statusText = 'Ready';
            break;
        case 'error':
            statusIcon = <ExclamationCircleIcon className="w-5 h-5 text-red-500" />;
            statusColor = 'text-red-600';
            statusText = file.error || 'Failed';
            break;
        case 'duplicate':
            statusIcon = <DocumentDuplicateIcon className="w-5 h-5 text-yellow-500" />;
            statusColor = 'text-yellow-600';
            statusText = 'Duplicate';
            break;
        default:
             statusIcon = null;
             statusColor = 'text-gray-400';
             statusText = 'Pending';
    }

    return (
        <li className="flex items-center justify-between bg-white p-3 rounded-lg border border-gray-200 text-sm">
            <div className="flex items-center gap-3 truncate">
                <DocumentTextIcon className="w-5 h-5 text-gray-400 flex-shrink-0" />
                <span className="font-medium text-gray-800 truncate" title={file.name}>{file.name}</span>
            </div>
             <div className="flex items-center gap-3 ml-2">
                <div className="flex items-center gap-1.5" title={statusText}>
                    {statusIcon}
                    <span className={`hidden sm:block text-xs font-medium truncate ${statusColor}`}>{statusText}</span>
                </div>
                <button onClick={onRemove} disabled={isProcessing} className="text-gray-400 hover:text-red-500 disabled:opacity-50 flex-shrink-0">
                    <XMarkIcon className="w-5 h-5" />
                </button>
            </div>
        </li>
    );
};

// ============================================================================
// Dashboard Components (New Redesign)
// ============================================================================

const DashboardCard: React.FC<{ title: string; children: React.ReactNode; className?: string }> = ({ title, children, className = '' }) => (
    <div className={`bg-gray-200/50 p-1.5 rounded-3xl ${className}`}>
        <div className="bg-white p-5 rounded-2xl h-full flex flex-col">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-base font-semibold text-gray-700">{title}</h3>
                <button className="text-gray-400 hover:text-gray-600">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M6 10a2 2 0 11-4 0 2 2 0 014 0zM12 10a2 2 0 11-4 0 2 2 0 014 0zM16 12a2 2 0 100-4 2 2 0 000 4z" />
                    </svg>
                </button>
            </div>
            <div className="flex-grow">
                {children}
            </div>
        </div>
    </div>
);

const DashboardHeader: React.FC<{ onReset: () => void; candidateCount: number }> = ({ onReset, candidateCount }) => (
    <header className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
            <h1 className="text-3xl font-semibold text-gray-800">Analysis Results</h1>
            <p className="text-gray-500 mt-1">Found {candidateCount} candidates matching your criteria.</p>
        </div>
        <button
            onClick={onReset}
            className="bg-white hover:bg-gray-50 text-gray-700 font-medium py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors border border-gray-300 shadow-sm"
        >
            <ArrowPathIcon className="w-5 h-5" />
            <span>Start New Analysis</span>
        </button>
    </header>
);

const FairnessScoreCard: React.FC<{ fairnessMetrics: FairnessMetrics }> = ({ fairnessMetrics }) => {
    const score = fairnessMetrics.fairness_score;
    const data = [{ value: score }, { value: 100 - score }];
    const scoreColor = score > 75 ? '#10B981' : score > 50 ? '#F59E0B' : '#EF4444';

    return (
        <DashboardCard title="Overall Fairness Score">
            <div className="h-full relative flex items-center justify-center">
                <ResponsiveContainer width="100%" height={180}>
                    <PieChart>
                        <Pie data={data} cx="50%" cy="50%" dataKey="value" innerRadius={60} outerRadius={75} startAngle={90} endAngle={-270}>
                            <Cell fill={scoreColor} stroke={scoreColor} />
                            <Cell fill="#F3F4F6" stroke="#F3F4F6" />
                        </Pie>
                    </PieChart>
                </ResponsiveContainer>
                <div className="absolute flex flex-col items-center justify-center pointer-events-none">
                    <span className="text-5xl font-bold text-gray-800">{score.toFixed(0)}</span>
                    <span className="text-sm font-medium text-gray-500">out of 100</span>
                </div>
            </div>
        </DashboardCard>
    );
};

const BiasAnalyticsCard: React.FC<{ candidates: Candidate[] }> = ({ candidates }) => {
    const biasDistribution = useMemo(() => {
        const counts = candidates.reduce((acc, c) => {
            const level = c.biasRiskLevel || 'UNKNOWN';
            acc[level] = (acc[level] || 0) + 1;
            return acc;
        }, {} as Record<string, number>);
        
        const orderedLevels = ['HIGH', 'MEDIUM', 'LOW', 'NONE', 'UNKNOWN'];
        return orderedLevels.filter(level => counts[level] > 0).map(level => ({
            name: level.charAt(0) + level.slice(1).toLowerCase(),
            value: counts[level]
        }));
    }, [candidates]);

    const BIAS_COLORS: Record<string, string> = { 'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#22c55e', 'None': '#6b7280', 'Unknown': '#9ca3af' };

    return (
        <DashboardCard title="Bias Risk Distribution">
            {biasDistribution.length > 0 ? (
                <div className="w-full h-full min-h-[180px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie data={biasDistribution} cx="50%" cy="50%" innerRadius={50} outerRadius={70} fill="#8884d8" paddingAngle={5} dataKey="value">
                                {biasDistribution.map((entry) => ( <Cell key={`cell-${entry.name}`} fill={BIAS_COLORS[entry.name]} /> ))}
                            </Pie>
                            <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '0.75rem' }} />
                            <Legend iconType="circle" wrapperStyle={{fontSize: "12px"}}/>
                        </PieChart>
                    </ResponsiveContainer>
                </div>
            ) : (
                 <div className="h-full flex items-center justify-center"><p className="text-sm text-gray-500">No bias data to display.</p></div>
            )}
        </DashboardCard>
    );
};

const TopCandidatesCard: React.FC<{candidates: Candidate[]}> = ({ candidates }) => {
     const topCandidatesData = useMemo(() => {
        return candidates.slice(0, 5).map(c => ({
            name: `Cand. ${c.id}`,
            score: c.overallScore,
        })).reverse();
    }, [candidates]);
    
    return (
        <DashboardCard title="Top 5 Candidates">
             <div className="w-full h-full min-h-[180px]">
                 <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={topCandidatesData} layout="vertical" margin={{ top: 0, right: 15, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e5e7eb" />
                        <XAxis type="number" domain={[0, 100]} stroke="#9ca3af" fontSize={12} />
                        <YAxis type="category" dataKey="name" stroke="#9ca3af" fontSize={12} width={55} />
                        <Tooltip cursor={{fill: '#f9fafb'}} contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '0.75rem' }}/>
                        <Bar dataKey="score" barSize={15} radius={[0, 4, 4, 0]}>
                             {topCandidatesData.map((entry) => {
                                const color = entry.score > 75 ? '#10B981' : entry.score > 50 ? '#F59E0B' : '#EF4444';
                                return <Cell key={`cell-${entry.name}`} fill={color} />;
                            })}
                        </Bar>
                    </BarChart>
                 </ResponsiveContainer>
             </div>
        </DashboardCard>
    );
};

const FairnessDetailsCard: React.FC<{ fairnessMetrics: FairnessMetrics }> = ({ fairnessMetrics }) => {
    const metrics = [
        { name: 'Disparate Impact', value: fairnessMetrics.disparate_impact, desc: "Ratio of selection rates between groups. Closer to 1.0 is better." },
        { name: 'Equal Opportunity', value: fairnessMetrics.equal_opportunity, desc: "Difference in true positive rates. Closer to 0 is better." },
        { name: 'Statistical Parity', value: fairnessMetrics.statistical_parity_difference, desc: "Difference in selection rates. Closer to 0 is better." },
    ];
    return (
        <DashboardCard title="Fairness Details">
            <div className="h-full space-y-4 pt-2">
                {metrics.map(m => (
                    <div key={m.name} className="flex justify-between items-center text-sm" title={m.desc}>
                        <div className="flex items-center gap-1.5">
                            <span className="text-gray-600">{m.name}</span>
                            <InformationCircleIcon className="w-4 h-4 text-gray-400" />
                        </div>
                        <span className="font-semibold text-gray-800 bg-gray-100 px-2 py-0.5 rounded">{m.value !== null ? m.value.toFixed(2) : 'N/A'}</span>
                    </div>
                ))}
            </div>
        </DashboardCard>
    );
};

const CandidateRankingsTable: React.FC<{ candidates: Candidate[] }> = ({ candidates }) => {
    const [expandedId, setExpandedId] = useState<number | null>(candidates.length > 0 ? candidates[0].id : null);
    
    return (
        <div className="bg-white rounded-2xl border border-gray-200/80 shadow-sm">
            <div className="p-5 border-b border-gray-200 flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-800">Candidate Rankings</h3>
                <span className="text-sm font-medium text-gray-500 bg-gray-100 px-3 py-1 rounded-full">Sorted by Overall Score</span>
            </div>
            <div className="overflow-x-auto">
                <div className="min-w-full divide-y divide-gray-200">
                    {candidates.length > 0 ? candidates.map((candidate, index) => (
                         <CandidateTableRow 
                            key={candidate.id} 
                            candidate={candidate} 
                            rank={index + 1} 
                            isExpanded={expandedId === candidate.id}
                            onToggle={() => setExpandedId(expandedId === candidate.id ? null : candidate.id)}
                        />
                    )) : (
                        <p className="p-6 text-center text-gray-500">No candidates to display.</p>
                    )}
                </div>
            </div>
        </div>
    );
};

const CandidateTableRow: React.FC<{ candidate: Candidate, rank: number, isExpanded: boolean, onToggle: () => void }> = ({ candidate, rank, isExpanded, onToggle }) => {
    const scoreColor = candidate.overallScore > 75 ? 'text-green-600 bg-green-50' : candidate.overallScore > 50 ? 'text-yellow-600 bg-yellow-50' : 'text-red-600 bg-red-50';
    const biasRiskMap: Record<string, string> = {
        'HIGH': 'bg-red-500', 'MEDIUM': 'bg-yellow-500', 'LOW': 'bg-green-500', 'NONE': 'bg-gray-400', 'UNKNOWN': 'bg-gray-300',
    };

    return (
        <div className="w-full">
            <button onClick={onToggle} className="w-full grid grid-cols-12 gap-4 px-5 py-3 text-sm text-left items-center hover:bg-gray-50 transition-colors">
                 <div className="col-span-1 font-semibold text-gray-500 text-lg">#{rank}</div>
                 <div className="col-span-5 md:col-span-4 flex items-center gap-3">
                    <div title={`Bias Risk: ${candidate.biasRiskLevel}`} className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${biasRiskMap[candidate.biasRiskLevel]}`}></div>
                    <span className="font-semibold text-gray-800">Candidate {candidate.id}</span>
                 </div>
                 <p className="hidden md:block col-span-5 text-gray-500 italic truncate">"{candidate.professionalSummary}"</p>
                 <div className={`col-span-4 md:col-span-1 text-base font-bold rounded-md px-2 py-1 text-center ${scoreColor}`}>{candidate.overallScore}</div>
                 <div className="col-span-2 md:col-span-1 flex justify-end">
                     <ChevronDownIcon className={`w-5 h-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                 </div>
            </button>
            {isExpanded && <CandidateDetailView candidate={candidate} />}
        </div>
    );
};

const ResultsDashboard: React.FC<{ result: AnalysisResponse; onReset: () => void; }> = ({ result, onReset }) => {
  const { candidates, fairnessMetrics } = result;

  return (
    <div className="min-h-screen">
      <div className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8">
        <DashboardHeader onReset={onReset} candidateCount={candidates.length} />
        <main className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <FairnessScoreCard fairnessMetrics={fairnessMetrics} />
            <BiasAnalyticsCard candidates={candidates} />
            <TopCandidatesCard candidates={candidates} />
            <FairnessDetailsCard fairnessMetrics={fairnessMetrics} />
          </div>
          <div className="mt-6">
            <CandidateRankingsTable candidates={candidates}/>
          </div>
        </main>
      </div>
    </div>
  );
};

const DetailSection: React.FC<{title: string; icon: React.ReactNode; children: React.ReactNode}> = ({title, icon, children}) => (
    <div>
        <h5 className="font-semibold mb-3 flex items-center gap-2 text-gray-800">
            {icon} {title}
        </h5>
        {children}
    </div>
);

const ProgressBar: React.FC<{label: string; value: number}> = ({label, value}) => (
    <div>
        <div className="flex justify-between mb-1 text-xs font-medium text-gray-600">
            <span>{label}</span>
            <span>{value.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-1.5">
            <div className="bg-indigo-500 h-1.5 rounded-full" style={{width: `${value}%`}}></div>
        </div>
    </div>
);

const CandidateDetailView: React.FC<{ candidate: Candidate }> = ({ candidate }) => (
    <div className="bg-gray-50 border-t border-b border-gray-200 p-6 space-y-8 text-sm">
        <DetailSection title="Overall Justification" icon={<LightBulbIcon className="w-4 h-4 text-yellow-500" />}>
            <p className="text-gray-600">{candidate.justification}</p>
        </DetailSection>

        <div className="grid md:grid-cols-2 gap-8">
            <DetailSection title="Skills Match" icon={<StarIcon className="w-4 h-4 text-yellow-500" />}>
                <div className="flex flex-wrap gap-2">
                    {candidate.skillsMatch.map(skill => <SkillTag key={skill.name} skill={skill} />)}
                     {candidate.skillsMatch.length === 0 && <p className="text-gray-500 text-xs">No specific skills matched from the job description.</p>}
                </div>
            </DetailSection>

            <DetailSection title="Semantic Match Analysis" icon={<LinkIcon className="w-4 h-4 text-indigo-500" />}>
                 <div className="space-y-3">
                    <ProgressBar label="Keyword Overlap" value={candidate.semanticMetrics.keyword_overlap * 100} />
                    <ProgressBar label="Contextual Similarity" value={candidate.semanticMetrics.cosine_similarity * 100} />
                 </div>
            </DetailSection>
        </div>

        <DetailSection title="Experience Analysis" icon={<BriefcaseIcon className="w-4 h-4 text-gray-700" />}>
            <p className="text-gray-600">{candidate.experienceMatch}</p>
        </DetailSection>

        <div className="grid md:grid-cols-2 gap-8">
             <DetailSection title="Education & Certifications" icon={<AcademicCapIcon className="w-4 h-4 text-purple-500" />}>
                 <p className="text-gray-600 mb-2"><span className="font-semibold text-gray-800">Level:</span> {candidate.educationLevel}</p>
                 {candidate.certifications.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                        {candidate.certifications.map(cert => (
                            <span key={cert} className="bg-purple-100 text-purple-700 text-xs font-medium px-2.5 py-1 rounded-full">{cert}</span>
                        ))}
                    </div>
                 )}
             </DetailSection>
             <DetailSection title="Fairness & Bias Report" icon={<ShieldCheckIcon className="w-4 h-4 text-green-500" />}>
                <p className="text-gray-600"><span className="font-semibold text-gray-800">Bias Risk Level:</span> {candidate.biasRiskLevel}</p>
                {candidate.biasIndicators.length > 0 && (
                    <p className="text-gray-600 mt-1"><span className="font-semibold text-gray-800">Categories Masked:</span> {candidate.biasIndicators.join(', ')}</p>
                )}
             </DetailSection>
        </div>
        
         {candidate.warnings.length > 0 && (
            <DetailSection title="Warnings" icon={<ExclamationTriangleIcon className="w-4 h-4 text-yellow-500" />}>
                <ul className="list-disc list-inside space-y-1 text-yellow-700">
                   {candidate.warnings.map(warning => <li key={warning}>{warning}</li>)}
                </ul>
            </DetailSection>
        )}
    </div>
);

const SkillTag: React.FC<{ skill: Skill }> = ({ skill }) => {
    const baseClasses = "flex items-center gap-1.5 text-xs font-medium py-1 px-2.5 rounded-full w-fit";
    const mentionedClasses = "bg-green-100 text-green-800";
    const notMentionedClasses = "bg-red-100 text-red-800";

    return (
        <span className={`${baseClasses} ${skill.mentioned ? mentionedClasses : notMentionedClasses}`}>
            {skill.mentioned ? <CheckCircleIcon className="w-3.5 h-3.5" /> : <XCircleIcon className="w-3.5 h-3.5" />}
            {skill.name}
        </span>
    );
};


const ProjectInfoModal: React.FC<{ isOpen: boolean; onClose: () => void; }> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div 
        className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] flex flex-col transition-all duration-300 transform scale-95 opacity-0 animate-fade-in-scale border border-gray-200"
        onClick={(e) => e.stopPropagation()}
        style={{ animationFillMode: 'forwards' }}
      >
        <header className="flex items-center justify-between p-4 border-b border-gray-200 sticky top-0 bg-white/80 backdrop-blur-lg">
          <h2 className="text-xl font-bold text-gray-800">About FairHirePro</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-indigo-600 transition-colors">
            <XMarkIcon className="w-6 h-6" />
          </button>
        </header>

        <div className="overflow-y-auto p-6 space-y-6">
            <Section title="Group Members"><p>MANYA BAJAJ (23215031) | RADHIKA (23215045)</p></Section>
            <div className="grid md:grid-cols-2 gap-6">
                <Section title="Mission"><p>CHRIST is a nurturing ground for an individual’s holistic development to make effective contribution to the society in a dynamic environment.</p></Section>
                <Section title="Vision & Core Values"><p>Excellence and Service, Faith in God, Moral Uprightness, Love of Fellow Beings, Social Responsibility, Pursuit of Excellence.</p></Section>
            </div>
            <Section title="Introduction: FairHire Pro"><p className="font-semibold">A Web-Based Unbiased Resume Screening and Ranking System.</p></Section>
            <Section title="Problem Statement"><ul className="list-disc list-inside space-y-1"><li>Manual resume screening is time-consuming.</li><li>Screening processes are prone to unconscious bias.</li><li>Traditional ATS lacks transparency and ethical data handling.</li><li>Need for a fair, explainable, and auditable recruitment system.</li></ul></Section>
            <Section title="Significance of the Project"><ul className="list-disc list-inside space-y-1"><li>Streamlines hiring for HR teams and recruiters.</li><li>Promotes diversity, equity, and inclusion (DEI).</li><li>Provides explainable rankings and transparency.</li><li>Aligns with global trends in ethical AI & HR Tech.</li></ul></Section>
        </div>
      </div>
      <style>{`
        @keyframes fade-in-scale { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }
        .animate-fade-in-scale { animation: fade-in-scale 0.3s cubic-bezier(0.16, 1, 0.3, 1); }
      `}</style>
    </div>
  );
};

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div>
    <h3 className="text-lg font-semibold text-gray-800 border-b border-gray-200 pb-2 mb-3">{title}</h3>
    <div className="text-gray-600 space-y-2">{children}</div>
  </div>
);


// ============================================================================
// Main App Component
// ============================================================================

const App: React.FC = () => {
  const [jobDescription, setJobDescription] = useState<string>('');
  const [resumeFiles, setResumeFiles] = useState<ResumeFile[]>([]);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isInfoModalOpen, setInfoModalOpen] = useState<boolean>(false);
  const [gdprConsent, setGdprConsent] = useState<boolean>(false);

  const isResultsView = analysisResult !== null;

  useEffect(() => {
    // Dynamically change body style based on the view
    document.body.style.backgroundColor = isResultsView ? '#E9EDF0' : '';
    document.body.style.backgroundImage = isResultsView ? 'none' : '';
    
    // Cleanup function to reset body style when component unmounts
    return () => {
        document.body.style.backgroundColor = '';
        document.body.style.backgroundImage = '';
    };
  }, [isResultsView]);


  const processFiles = useCallback(async (files: FileList) => {
    const newFiles = Array.from(files);
    const newEntries: ResumeFile[] = newFiles.map(file => ({ file, name: file.name, status: 'parsing', content: null, hash: null }));
    const currentHashes = new Set<string>(resumeFiles.map(f => f.hash).filter(Boolean) as string[]);
    
    setResumeFiles(prev => [...prev, ...newEntries]);

    const settledPromises = await Promise.allSettled(newFiles.map(file => parseResumeFile(file)));

    setResumeFiles(prev => {
        const resultsMap = new Map<File, { status: 'fulfilled' | 'rejected', value?: any, reason?: any }>();
        settledPromises.forEach((result, index) => resultsMap.set(newFiles[index], result));

        return prev.map(fileState => {
            const result = resultsMap.get(fileState.file);
            if (!result) return fileState;

            if (result.status === 'fulfilled') {
                const { content, hash } = result.value;
                if (currentHashes.has(hash)) {
                    return { ...fileState, status: 'duplicate', hash };
                }
                currentHashes.add(hash);
                return { ...fileState, status: 'success', content, hash };
            }
            const errorMessage = result.reason instanceof Error ? result.reason.message : 'Unknown parsing error';
            return { ...fileState, status: 'error', error: errorMessage };
        });
    });
  }, [resumeFiles]);

  const handleAnalyzeClick = async () => {
    const validResumes = resumeFiles.filter(f => f.status === 'success' && f.content);
    if (!jobDescription.trim() || validResumes.length === 0 || !gdprConsent) {
      setError("Please provide a job description, at least one valid resume, and consent to the terms.");
      return;
    }
    setError(null);
    setIsProcessing(true);
    try {
      const candidates = await analyzeResumes(jobDescription, validResumes);
      const fairnessMetrics = calculateFrontendFairnessMetrics(candidates);
      
      const fullResult: AnalysisResponse = {
          candidates,
          fairnessMetrics,
          metadata: {}, // Add empty metadata to satisfy the type
      };
      setAnalysisResult(fullResult);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An unknown error occurred.";
      setError(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setJobDescription('');
    setResumeFiles([]);
    setAnalysisResult(null);
    setError(null);
    setIsProcessing(false);
    setGdprConsent(false);
  };
  
  const removeFile = (fileToRemove: ResumeFile) => {
    setResumeFiles(prevFiles => prevFiles.filter(f => f !== fileToRemove));
  };

  const successfullyUploadedResumes = resumeFiles.filter(f => f.status === 'success');
  const canAnalyze = jobDescription.trim() && successfullyUploadedResumes.length > 0 && !isProcessing && gdprConsent;

  const MobileHeader: React.FC<{ onInfoClick: () => void }> = ({ onInfoClick }) => (
    <div className="lg:hidden w-full flex justify-between items-center mb-4 px-2">
        <div className="flex items-center gap-2">
            <BriefcaseIcon className="h-6 w-6 text-indigo-600" />
            <h1 className="text-xl font-bold text-gray-800">FairHirePro</h1>
        </div>
        <button onClick={onInfoClick} className="text-gray-500 hover:text-indigo-600">
            <InformationCircleIcon className="w-6 h-6" />
        </button>
    </div>
  );
  
  if (isResultsView) {
      return <ResultsDashboard result={analysisResult} onReset={handleReset} />;
  }

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center p-4 sm:p-6 lg:p-8">
      
      {isProcessing && (
           <div className="flex flex-col items-center justify-center text-center p-8 bg-white/50 backdrop-blur-md rounded-2xl shadow-lg">
              <SpinnerIcon className="h-12 w-12 text-indigo-600 animate-spin" />
              <h2 className="text-2xl font-bold mt-6 text-gray-800">Analyzing Candidates...</h2>
              <p className="text-gray-600 mt-2 max-w-md">Our AI is carefully reviewing each resume for skills, experience, and fairness. This may take a moment.</p>
          </div>
      )}

      {!isProcessing && (
        <div className="w-full max-w-6xl animate-fade-in-scale">
          <MobileHeader onInfoClick={() => setInfoModalOpen(true)} />
          <div className="w-full bg-white/80 backdrop-blur-lg rounded-2xl shadow-2xl flex flex-col lg:flex-row overflow-hidden border border-white/50">
            <BrandPanel onInfoClick={() => setInfoModalOpen(true)} />
            <div className="w-full lg:w-2/3 p-6 sm:p-8 space-y-6 flex flex-col">
              <div className="flex-grow space-y-6">
                <JobDescriptionInput jobDescription={jobDescription} setJobDescription={setJobDescription} isProcessing={isProcessing} />
                <ResumeInputSection onFilesAdded={processFiles} isProcessing={isProcessing}/>
                {resumeFiles.length > 0 && (
                  <div className="space-y-2">
                      <label className="text-sm font-semibold text-gray-700">Uploaded Files ({resumeFiles.length})</label>
                      <ul className="space-y-2 max-h-48 overflow-y-auto pr-2">
                          {resumeFiles.map((file, index) => (
                              <FileStatusItem key={`${file.name}-${index}`} file={file} onRemove={() => removeFile(file)} isProcessing={isProcessing} />
                          ))}
                      </ul>
                  </div>
                )}
              </div>
              <div className="space-y-4 text-center pt-4 border-t border-gray-200">
                  <div className="flex items-center justify-center gap-3">
                      <input type="checkbox" id="gdpr-consent" checked={gdprConsent} onChange={(e) => setGdprConsent(e.target.checked)} className="h-4 w-4 rounded border-gray-400 text-indigo-600 focus:ring-indigo-500" disabled={isProcessing}/>
                      <label htmlFor="gdpr-consent" className="text-sm text-gray-600">I consent to process the uploaded resume data for analysis.</label>
                  </div>
                  <button onClick={handleAnalyzeClick} disabled={!canAnalyze} className="bg-indigo-600 text-white font-bold py-3 px-8 rounded-lg shadow-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed disabled:shadow-none transform hover:scale-105 transition-all duration-300 flex items-center gap-2 mx-auto w-full sm:w-auto justify-center">
                      <SparklesIcon className="w-5 h-5" />
                      Analyze {successfullyUploadedResumes.length > 0 ? `${successfullyUploadedResumes.length} Candidate(s)` : ''}
                  </button>
                  {error && <p className="text-red-600 text-sm mt-2">{error}</p>}
              </div>
            </div>
          </div>
        </div>
      )}

      <ProjectInfoModal isOpen={isInfoModalOpen} onClose={() => setInfoModalOpen(false)} />
    </div>
  );
};

export default App;
