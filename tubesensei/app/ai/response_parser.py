"""
Parse and validate AI responses with error handling.

This module provides:
- JSON response parsing with validation
- Error recovery for malformed responses
- Type checking and data extraction
- Fallback parsing strategies
"""

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ParsedIdea:
    """Structured idea extracted from AI response."""
    title: str
    description: str
    category: str
    target_market: str = ""
    value_proposition: str = ""
    complexity_score: int = 5
    confidence: float = 0.5
    source_context: str = ""

@dataclass
class ParsedVideoFilter:
    """Video filtering result from AI analysis."""
    is_valuable: bool
    confidence_score: float
    reasoning: str = ""
    detected_topics: List[str] = None
    predicted_idea_count: int = 0
    
    def __post_init__(self):
        if self.detected_topics is None:
            self.detected_topics = []

@dataclass
class ParsedQualityAssessment:
    """Quality assessment result from AI analysis."""
    quality_score: float
    viability_scores: Dict[str, float] = None
    strengths: List[str] = None
    weaknesses: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.viability_scores is None:
            self.viability_scores = {}
        if self.strengths is None:
            self.strengths = []
        if self.weaknesses is None:
            self.weaknesses = []
        if self.recommendations is None:
            self.recommendations = []

class ResponseParser:
    """Parse and validate LLM responses."""
    
    @staticmethod
    def parse_json_response(
        response: str,
        required_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Parse JSON response with error recovery.
        
        Args:
            response: Raw LLM response text
            required_fields: Fields that must be present
            
        Returns:
            Parsed JSON as dictionary
        """
        # Try direct JSON parsing
        try:
            # Remove any markdown formatting
            clean_response = response.strip()
            if "```json" in response:
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    clean_response = json_match.group(1).strip()
            elif "```" in response:
                json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    clean_response = json_match.group(1).strip()
            
            data = json.loads(clean_response)
            
            # Validate required fields
            if required_fields:
                for field in required_fields:
                    if field not in data:
                        raise ValueError(f"Missing required field: {field}")
            
            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parsing failed: {e}")
            
            # Try to extract JSON-like content
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Fallback: create minimal valid response
            if required_fields:
                return {field: None for field in required_fields}
            return {}
    
    @staticmethod
    def parse_video_filter_response(response: str) -> ParsedVideoFilter:
        """Parse video filtering AI response."""
        result = ResponseParser.parse_json_response(
            response,
            required_fields=["is_valuable", "confidence_score"]
        )
        
        # Ensure proper types with validation
        is_valuable = bool(result.get("is_valuable", False))
        confidence_score = min(1.0, max(0.0, float(result.get("confidence_score", 0.5))))
        reasoning = str(result.get("reasoning", ""))[:1000]  # Limit length
        detected_topics = result.get("detected_topics", [])
        if not isinstance(detected_topics, list):
            detected_topics = []
        predicted_idea_count = max(0, int(result.get("predicted_idea_count", 0)))
        
        return ParsedVideoFilter(
            is_valuable=is_valuable,
            confidence_score=confidence_score,
            reasoning=reasoning,
            detected_topics=detected_topics,
            predicted_idea_count=predicted_idea_count
        )
    
    @staticmethod
    def parse_idea_extraction_response(response: str) -> List[ParsedIdea]:
        """Parse idea extraction AI response."""
        result = ResponseParser.parse_json_response(
            response,
            required_fields=["ideas"]
        )
        
        ideas = []
        for idea_data in result.get("ideas", []):
            try:
                # Validate and clean data with length limits
                title = str(idea_data.get("title", "")).strip()[:100]
                description = str(idea_data.get("description", "")).strip()[:500]
                category = str(idea_data.get("category", "Other")).strip()[:50]
                target_market = str(idea_data.get("target_market", "")).strip()[:200]
                value_proposition = str(idea_data.get("value_proposition", "")).strip()[:300]
                
                # Validate scores with proper ranges
                complexity_score = min(10, max(1, int(idea_data.get("complexity_score", 5))))
                confidence = min(1.0, max(0.0, float(idea_data.get("confidence", 0.5))))
                source_context = str(idea_data.get("source_context", "")).strip()[:500]
                
                idea = ParsedIdea(
                    title=title,
                    description=description,
                    category=category,
                    target_market=target_market,
                    value_proposition=value_proposition,
                    complexity_score=complexity_score,
                    confidence=confidence,
                    source_context=source_context
                )
                
                # Skip invalid ideas (must have title and description)
                if idea.title and idea.description:
                    ideas.append(idea)
                    
            except Exception as e:
                logger.warning(f"Failed to parse idea: {e}")
                continue
        
        return ideas
    
    @staticmethod
    def parse_idea_categorization_response(response: str) -> Dict[str, Any]:
        """Parse idea categorization AI response."""
        result = ResponseParser.parse_json_response(
            response,
            required_fields=["categorized_ideas"]
        )
        
        # Validate categorized ideas structure
        categorized_ideas = result.get("categorized_ideas", {})
        if not isinstance(categorized_ideas, dict):
            categorized_ideas = {}
        
        # Clean and validate each category
        cleaned_categories = {}
        for category, ideas in categorized_ideas.items():
            category_name = str(category).strip()[:50]
            if isinstance(ideas, list):
                cleaned_categories[category_name] = ideas
            else:
                cleaned_categories[category_name] = []
        
        result["categorized_ideas"] = cleaned_categories
        result["category_summary"] = str(result.get("category_summary", ""))[:1000]
        
        return result
    
    @staticmethod
    def parse_quality_assessment_response(response: str) -> ParsedQualityAssessment:
        """Parse quality assessment AI response."""
        result = ResponseParser.parse_json_response(
            response,
            required_fields=["quality_score"]
        )
        
        # Ensure score is in valid range
        quality_score = min(1.0, max(0.0, float(result.get("quality_score", 0.5))))
        
        # Validate viability scores
        viability_scores = result.get("viability_assessment", {})
        if isinstance(viability_scores, dict):
            cleaned_scores = {}
            for key, score in viability_scores.items():
                try:
                    cleaned_scores[str(key)] = min(1.0, max(0.0, float(score)))
                except (ValueError, TypeError):
                    cleaned_scores[str(key)] = 0.5
            viability_scores = cleaned_scores
        else:
            viability_scores = {}
        
        # Validate lists with length limits
        strengths = result.get("strengths", [])
        if isinstance(strengths, list):
            strengths = [str(s)[:200] for s in strengths[:10]]  # Max 10 items, 200 chars each
        else:
            strengths = []
        
        weaknesses = result.get("weaknesses", [])
        if isinstance(weaknesses, list):
            weaknesses = [str(w)[:200] for w in weaknesses[:10]]
        else:
            weaknesses = []
        
        recommendations = result.get("recommendations", [])
        if isinstance(recommendations, list):
            recommendations = [str(r)[:300] for r in recommendations[:10]]
        else:
            recommendations = []
        
        return ParsedQualityAssessment(
            quality_score=quality_score,
            viability_scores=viability_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )
    
    @staticmethod
    def parse_summary_generation_response(response: str) -> Dict[str, Any]:
        """Parse summary generation AI response."""
        result = ResponseParser.parse_json_response(
            response,
            required_fields=["summary"]
        )
        
        # Validate and clean summary data
        summary = str(result.get("summary", "")).strip()[:2000]  # Limit summary length
        key_points = result.get("key_points", [])
        
        if isinstance(key_points, list):
            # Clean and limit key points
            key_points = [str(point)[:300] for point in key_points[:20]]  # Max 20 points, 300 chars each
        else:
            key_points = []
        
        # Optional fields
        insights = result.get("insights", [])
        if isinstance(insights, list):
            insights = [str(insight)[:400] for insight in insights[:10]]
        else:
            insights = []
        
        return {
            "summary": summary,
            "key_points": key_points,
            "insights": insights,
            "word_count": len(summary.split()) if summary else 0
        }
    
    @staticmethod
    def extract_structured_data(
        text: str,
        patterns: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Extract structured data using regex patterns as fallback.
        
        Args:
            text: Text to extract from
            patterns: Dictionary of field_name: regex_pattern
            
        Returns:
            Extracted data dictionary
        """
        results = {}
        
        for field, pattern in patterns.items():
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if match:
                    # Take the first capturing group if available, otherwise the whole match
                    if match.groups():
                        results[field] = match.group(1).strip()
                    else:
                        results[field] = match.group().strip()
                else:
                    results[field] = None
            except re.error as e:
                logger.warning(f"Invalid regex pattern for field '{field}': {e}")
                results[field] = None
        
        return results