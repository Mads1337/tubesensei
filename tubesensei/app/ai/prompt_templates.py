"""
Centralized prompt templates for AI operations in TubeSensei.

This module provides:
- PromptType enum for different AI operations
- PromptTemplates class with system and user prompts
- Template variable validation
- Standardized prompt formatting
"""

import re
from enum import Enum
from typing import Dict, Any, Optional, Set


class PromptType(Enum):
    """Types of prompts for different processing stages."""
    VIDEO_FILTER = "video_filter"
    TOPIC_FILTER = "topic_filter"  # For topic-based campaign filtering
    IDEA_EXTRACTION = "idea_extraction"
    IDEA_CATEGORIZATION = "idea_categorization"
    QUALITY_ASSESSMENT = "quality_assessment"
    IDEA_CONSOLIDATION = "idea_consolidation"
    SUMMARY_GENERATION = "summary_generation"


class PromptTemplates:
    """Manages prompt templates for AI operations."""
    
    # System prompts for different tasks
    SYSTEM_PROMPTS = {
        PromptType.VIDEO_FILTER: """You are an expert content analyst specializing in identifying valuable business and technology content from YouTube videos. Your task is to evaluate whether videos contain actionable business ideas, innovative concepts, or valuable insights.""",

        PromptType.TOPIC_FILTER: """You are an expert content analyst specializing in determining YouTube video relevance to specific topics. Your task is to analyze video metadata (title, description, channel) and determine if the video is relevant to the given topic. Be thorough but fair - a video can be tangentially related and still be relevant. Focus on the core subject matter, not production quality or popularity.""",

        PromptType.IDEA_EXTRACTION: """You are a highly selective business analyst. Your job is to extract only genuinely distinct, specific, and implementable business ideas. You reject generic advice, vague concepts, and near-duplicates. Quality over quantity — returning 0 ideas is better than returning low-quality ones.""",
        
        PromptType.IDEA_CATEGORIZATION: """You are an expert at categorizing business ideas by industry, complexity, and potential. Provide structured categorization with clear reasoning.""",
        
        PromptType.QUALITY_ASSESSMENT: """You are a venture capitalist evaluating business ideas for viability and potential. Assess ideas based on market opportunity, feasibility, and uniqueness.""",
        
        PromptType.IDEA_CONSOLIDATION: """You are a deduplication and quality filter for business ideas. You merge semantically similar ideas into the strongest version and discard vague or generic entries. Be aggressive about merging — two ideas targeting the same market with the same mechanism are one idea.""",

        PromptType.SUMMARY_GENERATION: """You are an expert content summarizer specializing in business and technology content. Create concise, actionable summaries that capture the key insights and value propositions."""
    }
    
    # User prompt templates
    TEMPLATES = {
        PromptType.VIDEO_FILTER: """
Analyze this YouTube video metadata and determine if it likely contains valuable business ideas or insights.

Title: {title}
Description: {description}
Channel: {channel_name}
Duration: {duration_minutes} minutes
View Count: {view_count}
Published: {published_date}

Evaluate based on:
1. Title and description content relevance
2. Indicators of business/startup/technology focus
3. Educational or informative nature
4. Practical value for entrepreneurs

Respond in JSON format:
{{
    "is_valuable": true/false,
    "confidence_score": 0.0-1.0,
    "reasoning": "Brief explanation",
    "detected_topics": ["topic1", "topic2"],
    "predicted_idea_count": 0-10
}}
""",

        PromptType.TOPIC_FILTER: """
Analyze this YouTube video to determine if it is relevant to the specified topic.

Topic: {topic}

Video Title: {title}
Video Description: {description}
Channel Name: {channel_name}

Evaluate the video's relevance by considering:
1. Direct keyword matches between the topic and video metadata
2. Semantic similarity - does the video discuss the same concepts?
3. Contextual relevance - would someone interested in the topic find this valuable?
4. How closely does the content align with the core subject?

Alignment categories:
- "exact": Directly about the topic, primary focus
- "related": Strongly related, covers similar themes
- "tangential": Loosely connected, might be useful
- "unrelated": No meaningful connection to the topic

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{{
    "is_relevant": true or false,
    "relevance_score": 0.0 to 1.0,
    "reasoning": "Brief explanation of why the video is or isn't relevant (max 100 words)",
    "matched_keywords": ["keyword1", "keyword2"],
    "topic_alignment": "exact" or "related" or "tangential" or "unrelated"
}}
""",

        PromptType.IDEA_EXTRACTION: """
Extract only the most specific, unique, and implementable business ideas from this transcript. At most 5 per section. Most transcripts contain 0-3 genuinely distinct ideas.

Transcript:
{transcript}

Video Context:
- Title: {title}
- Channel: {channel_name}
- Duration: {duration_minutes} minutes

ACCEPT an idea only if it has ALL of these:
- A specific product, service, or business model (not just a category)
- A clear target market or customer segment
- A defined approach, mechanism, or value proposition
- Enough detail that someone could start building it

REJECT and do NOT include:
- Generic business advice ("use social media", "build an audience", "find a niche")
- Trend observations without a concrete business model ("AI is growing")
- Vague concepts ("some kind of marketplace")
- Slight variations of the same core idea (pick the best version only)
- Ideas the speaker mentions only in passing without substance

Confidence calibration:
- 0.9-1.0: Speaker explicitly described the idea with specifics (product, market, approach)
- 0.7-0.89: Idea is clearly implied with enough detail to act on
- Below 0.7: You are inferring or extrapolating beyond what was said — do NOT include these

For each idea provide:
1. Clear, concise title (max 100 chars)
2. Detailed description (200-500 chars)
3. Category (SaaS, E-commerce, Mobile App, Service, Platform, Hardware, Other)
4. Target market
5. Key value proposition
6. Estimated complexity (1-10 scale)
7. Confidence score (see calibration above)
8. Source context (direct quote or paraphrase from transcript)

Example of a well-extracted idea:
{{
    "title": "AI-Powered Invoice Reconciliation Tool",
    "description": "A SaaS platform that uses machine learning to automatically match invoices with purchase orders and bank statements, reducing manual accounting work by 80%. Integrates with QuickBooks, Xero, and SAP.",
    "category": "SaaS",
    "target_market": "SMBs and accounting departments at mid-market companies",
    "value_proposition": "Eliminates 10+ hours/week of manual reconciliation work per accountant",
    "complexity_score": 6,
    "confidence": 0.9,
    "source_context": "The host says: 'we spent 3 weeks building an invoice matcher and our clients saved 80% of their reconciliation time'"
}}

It is better to return 0 ideas than to return low-quality or duplicate ideas.

Respond in JSON format:
{{
    "ideas": [
        {{
            "title": "Idea title",
            "description": "Detailed description",
            "category": "Category",
            "target_market": "Target audience",
            "value_proposition": "Key value",
            "complexity_score": 1,
            "confidence": 0.0,
            "source_context": "Quote or context from transcript"
        }}
    ],
    "summary": "Brief summary of ideas found",
    "total_ideas": 0
}}
""",

        PromptType.IDEA_CATEGORIZATION: """
Categorize and enrich this business idea with additional metadata.

Idea Title: {title}
Idea Description: {description}
Source Video: {video_title}

Provide comprehensive categorization:
1. Primary Industry/Vertical
2. Business Model Type
3. Revenue Model
4. Market Size Estimate (Small/Medium/Large/Huge)
5. Time to Market (Weeks/Months/Years)
6. Required Skills/Expertise
7. Initial Investment Range
8. Competitive Landscape Assessment
9. Innovation Level (1-10)
10. Success Probability (Low/Medium/High)

Respond in JSON format:
{{
    "industry": "Primary industry",
    "business_model": "Type of business model",
    "revenue_model": "How it makes money",
    "market_size": "Size estimate",
    "time_to_market": "Development time",
    "required_skills": ["skill1", "skill2"],
    "investment_range": "$X - $Y",
    "competition_level": "Low/Medium/High",
    "innovation_score": 1-10,
    "success_probability": "Assessment",
    "additional_notes": "Any other insights"
}}
""",

        PromptType.QUALITY_ASSESSMENT: """
Assess the quality and viability of this extracted business idea.

Idea: {title}
Description: {description}
Category: {category}
Source: {source_context}

Evaluate the idea on these criteria:
1. Clarity and Specificity (Is it well-defined?)
2. Market Need (Does it solve a real problem?)
3. Feasibility (Can it be realistically implemented?)
4. Uniqueness (Is it differentiated from existing solutions?)
5. Scalability (Can it grow significantly?)
6. Revenue Potential (Is there a clear monetization path?)

Provide a detailed assessment with:
- Overall quality score (0.0-1.0)
- Strengths and weaknesses
- Recommended next steps
- Similar existing solutions (if any)
- Potential challenges

Respond in JSON format:
{{
    "quality_score": 0.0-1.0,
    "viability_assessment": {{
        "clarity": 0.0-1.0,
        "market_need": 0.0-1.0,
        "feasibility": 0.0-1.0,
        "uniqueness": 0.0-1.0,
        "scalability": 0.0-1.0,
        "revenue_potential": 0.0-1.0
    }},
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "recommended_actions": ["action1", "action2"],
    "similar_solutions": ["solution1", "solution2"],
    "main_challenges": ["challenge1", "challenge2"],
    "overall_recommendation": "Pursue/Consider/Skip"
}}
""",

        PromptType.IDEA_CONSOLIDATION: """
You are given a list of candidate business ideas extracted from a single video. Many may be duplicates, near-duplicates, or too vague to be useful.

Video: {video_title}

Candidate ideas (JSON):
{candidate_ideas_json}

Your tasks:
1. MERGE ideas that describe the same core concept (same market + same mechanism = one idea). Keep the best title, combine the strongest details from both descriptions.
2. DISCARD ideas that are generic advice, trend observations, or too vague to act on.
3. Return only genuinely distinct, specific, implementable ideas.

For each surviving idea, preserve all original fields (title, description, category, target_market, value_proposition, complexity_score, confidence, source_context). When merging, pick the higher confidence and combine source_context values.

Respond in JSON format:
{{
    "consolidated_ideas": [
        {{
            "title": "Idea title",
            "description": "Detailed description",
            "category": "Category",
            "target_market": "Target audience",
            "value_proposition": "Key value",
            "complexity_score": 1,
            "confidence": 0.0,
            "source_context": "Quote or context from transcript"
        }}
    ],
    "merge_log": ["Merged 'X' and 'Y' into 'Z'", "Discarded 'W' (too vague)"],
    "original_count": 0,
    "final_count": 0
}}
""",

        PromptType.SUMMARY_GENERATION: """
Generate a comprehensive summary of the business insights and ideas from this content.

Content Title: {title}
Content Source: {source}
Total Ideas Found: {idea_count}

Ideas Summary:
{ideas_summary}

Key Insights:
{key_insights}

Create a structured summary that includes:
1. Main themes and topics covered
2. Top 3-5 most valuable ideas
3. Overall assessment of content quality
4. Target audience recommendations
5. Key takeaways for entrepreneurs

Respond in JSON format:
{{
    "executive_summary": "Brief 2-3 sentence overview",
    "main_themes": ["theme1", "theme2", "theme3"],
    "top_ideas": [
        {{
            "title": "Idea title",
            "description": "Brief description",
            "value_score": 0.0-1.0
        }}
    ],
    "content_quality": {{
        "overall_score": 0.0-1.0,
        "depth": 0.0-1.0,
        "actionability": 0.0-1.0,
        "uniqueness": 0.0-1.0
    }},
    "target_audience": ["audience1", "audience2"],
    "key_takeaways": ["takeaway1", "takeaway2", "takeaway3"],
    "recommended_next_steps": ["step1", "step2"]
}}
"""
    }
    
    @classmethod
    def get_prompt(
        cls,
        prompt_type: PromptType,
        variables: Dict[str, Any],
        include_system: bool = True
    ) -> tuple[Optional[str], str]:
        """
        Get formatted prompt with variables substituted.
        
        Args:
            prompt_type: Type of prompt to retrieve
            variables: Variables to substitute in template
            include_system: Whether to include system prompt
            
        Returns:
            Tuple of (system_prompt, user_prompt)
            
        Raises:
            ValueError: If prompt type is unknown or required variables are missing
        """
        # Get template
        template = cls.TEMPLATES.get(prompt_type)
        if not template:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        # Validate variables before formatting
        cls.validate_variables(prompt_type, variables)
        
        # Format with variables
        try:
            user_prompt = template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")
        
        # Get system prompt if requested
        system_prompt = None
        if include_system:
            system_prompt = cls.SYSTEM_PROMPTS.get(prompt_type)
        
        return system_prompt, user_prompt
    
    @classmethod
    def validate_variables(cls, prompt_type: PromptType, variables: Dict[str, Any]) -> bool:
        """
        Validate that all required variables are provided.
        
        Args:
            prompt_type: Type of prompt to validate
            variables: Variables provided for template
            
        Returns:
            True if all required variables are present
            
        Raises:
            ValueError: If required variables are missing
        """
        template = cls.TEMPLATES.get(prompt_type, "")
        
        # Extract variable names from template using regex
        required_vars = set(re.findall(r'\{(\w+)\}', template))
        provided_vars = set(variables.keys())
        
        missing = required_vars - provided_vars
        if missing:
            raise ValueError(f"Missing required variables for {prompt_type.value}: {missing}")
        
        return True
    
    @classmethod
    def get_required_variables(cls, prompt_type: PromptType) -> Set[str]:
        """
        Get the set of required variables for a prompt type.
        
        Args:
            prompt_type: Type of prompt to analyze
            
        Returns:
            Set of required variable names
        """
        template = cls.TEMPLATES.get(prompt_type, "")
        return set(re.findall(r'\{(\w+)\}', template))
    
    @classmethod
    def list_prompt_types(cls) -> list[PromptType]:
        """
        Get list of all available prompt types.
        
        Returns:
            List of available PromptType enum values
        """
        return list(PromptType)
    
    @classmethod
    def get_prompt_info(cls, prompt_type: PromptType) -> Dict[str, Any]:
        """
        Get information about a prompt type including required variables.
        
        Args:
            prompt_type: Type of prompt to get info for
            
        Returns:
            Dictionary with prompt information
        """
        return {
            "type": prompt_type.value,
            "has_system_prompt": prompt_type in cls.SYSTEM_PROMPTS,
            "has_user_template": prompt_type in cls.TEMPLATES,
            "required_variables": list(cls.get_required_variables(prompt_type)),
            "system_prompt_preview": (
                cls.SYSTEM_PROMPTS.get(prompt_type, "")[:100] + "..." 
                if cls.SYSTEM_PROMPTS.get(prompt_type, "") 
                else None
            )
        }