"""
Tests for response_parser.py module.

Tests JSON parsing, response validation, data structure parsing,
and error recovery for malformed AI responses.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from tubesensei.app.ai.response_parser import (
    ResponseParser,
    ParsedIdea, 
    ParsedVideoFilter,
    ParsedQualityAssessment
)


class TestParsedDataClasses:
    """Test the parsed data classes."""
    
    def test_parsed_idea_instantiation(self):
        """Test ParsedIdea can be created with required fields."""
        idea = ParsedIdea(
            title="Test Idea",
            description="Test description",
            category="SaaS"
        )
        
        assert idea.title == "Test Idea"
        assert idea.description == "Test description"
        assert idea.category == "SaaS"
        assert idea.target_market == ""  # Default value
        assert idea.complexity_score == 5  # Default value
        assert idea.confidence == 0.5  # Default value
    
    def test_parsed_video_filter_instantiation(self):
        """Test ParsedVideoFilter with post_init."""
        filter_result = ParsedVideoFilter(
            is_valuable=True,
            confidence_score=0.8
        )
        
        assert filter_result.is_valuable is True
        assert filter_result.confidence_score == 0.8
        assert filter_result.detected_topics == []  # Set by post_init
        assert filter_result.predicted_idea_count == 0  # Default
    
    def test_parsed_video_filter_with_topics(self):
        """Test ParsedVideoFilter with provided topics."""
        topics = ["business", "technology"]
        filter_result = ParsedVideoFilter(
            is_valuable=True,
            confidence_score=0.9,
            detected_topics=topics
        )
        
        assert filter_result.detected_topics == topics
    
    def test_parsed_quality_assessment_instantiation(self):
        """Test ParsedQualityAssessment with post_init."""
        assessment = ParsedQualityAssessment(quality_score=0.7)
        
        assert assessment.quality_score == 0.7
        assert assessment.viability_scores == {}  # Set by post_init
        assert assessment.strengths == []  # Set by post_init
        assert assessment.weaknesses == []  # Set by post_init
        assert assessment.recommendations == []  # Set by post_init


class TestResponseParser:
    """Test ResponseParser static methods."""
    
    def test_parse_json_response_valid_json(self):
        """Test parsing valid JSON response."""
        json_response = '{"status": "success", "data": {"count": 5}}'
        
        result = ResponseParser.parse_json_response(json_response)
        
        assert result == {"status": "success", "data": {"count": 5}}
    
    def test_parse_json_response_markdown_wrapped_json(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        markdown_response = '''
        Here's the result:
        
        ```json
        {
            "is_valuable": true,
            "confidence_score": 0.85,
            "reasoning": "Contains valuable business insights"
        }
        ```
        '''
        
        result = ResponseParser.parse_json_response(markdown_response)
        
        expected = {
            "is_valuable": True,
            "confidence_score": 0.85,
            "reasoning": "Contains valuable business insights"
        }
        assert result == expected
    
    def test_parse_json_response_generic_code_block(self):
        """Test parsing JSON in generic code blocks."""
        code_block_response = '''
        ```
        {"test": "value", "number": 42}
        ```
        '''
        
        result = ResponseParser.parse_json_response(code_block_response)
        
        assert result == {"test": "value", "number": 42}
    
    def test_parse_json_response_malformed_json_fallback(self):
        """Test fallback behavior with malformed JSON."""
        malformed_response = 'This is not JSON at all!'
        
        result = ResponseParser.parse_json_response(malformed_response)
        
        assert result == {}  # Fallback to empty dict
    
    def test_parse_json_response_malformed_json_with_required_fields(self):
        """Test fallback behavior with required fields specified."""
        malformed_response = 'Not JSON'
        required_fields = ["field1", "field2"]
        
        result = ResponseParser.parse_json_response(
            malformed_response, 
            required_fields=required_fields
        )
        
        expected = {"field1": None, "field2": None}
        assert result == expected
    
    def test_parse_json_response_partial_json_extraction(self):
        """Test extraction of JSON-like content from mixed text."""
        mixed_response = '''
        Here's some text before the JSON.
        {"extracted": true, "value": 123}
        And some text after.
        '''
        
        result = ResponseParser.parse_json_response(mixed_response)
        
        assert result == {"extracted": True, "value": 123}
    
    def test_parse_json_response_required_fields_validation(self):
        """Test required fields validation."""
        json_response = '{"field1": "value1", "field2": "value2"}'
        required_fields = ["field1", "field2"]
        
        result = ResponseParser.parse_json_response(
            json_response, 
            required_fields=required_fields
        )
        
        assert result == {"field1": "value1", "field2": "value2"}
    
    def test_parse_json_response_missing_required_field_raises_error(self):
        """Test that missing required field raises ValueError."""
        json_response = '{"field1": "value1"}'
        required_fields = ["field1", "field2"]
        
        # Should fall back to providing None values for missing fields
        result = ResponseParser.parse_json_response(
            json_response,
            required_fields=required_fields
        )
        # The function should handle the ValueError internally and not raise it
        assert "field1" in result


class TestVideoFilterResponseParsing:
    """Test video filter response parsing."""
    
    def test_parse_video_filter_response_valid(self):
        """Test parsing valid video filter response."""
        response = '''
        {
            "is_valuable": true,
            "confidence_score": 0.85,
            "reasoning": "Strong business content indicators",
            "detected_topics": ["entrepreneurship", "startups"],
            "predicted_idea_count": 3
        }
        '''
        
        result = ResponseParser.parse_video_filter_response(response)
        
        assert isinstance(result, ParsedVideoFilter)
        assert result.is_valuable is True
        assert result.confidence_score == 0.85
        assert result.reasoning == "Strong business content indicators"
        assert result.detected_topics == ["entrepreneurship", "startups"]
        assert result.predicted_idea_count == 3
    
    def test_parse_video_filter_response_type_validation(self):
        """Test type validation and conversion in video filter parsing."""
        response = '''
        {
            "is_valuable": "true",
            "confidence_score": "0.95",
            "reasoning": 123,
            "detected_topics": "not a list",
            "predicted_idea_count": "5"
        }
        '''
        
        result = ResponseParser.parse_video_filter_response(response)
        
        assert result.is_valuable is True  # String converted to bool
        assert result.confidence_score == 0.95  # String converted to float
        assert result.reasoning == "123"  # Number converted to string
        assert result.detected_topics == []  # Invalid list becomes empty list
        assert result.predicted_idea_count == 5  # String converted to int
    
    def test_parse_video_filter_response_score_range_validation(self):
        """Test confidence score is clamped to valid range."""
        response = '''
        {
            "is_valuable": true,
            "confidence_score": 1.5,
            "predicted_idea_count": -2
        }
        '''
        
        result = ResponseParser.parse_video_filter_response(response)
        
        assert result.confidence_score == 1.0  # Clamped to max 1.0
        assert result.predicted_idea_count == 0  # Negative becomes 0
    
    def test_parse_video_filter_response_reasoning_length_limit(self):
        """Test reasoning field length limiting."""
        long_reasoning = "A" * 1500  # Longer than 1000 char limit
        response = f'''
        {{
            "is_valuable": true,
            "confidence_score": 0.8,
            "reasoning": "{long_reasoning}"
        }}
        '''
        
        result = ResponseParser.parse_video_filter_response(response)
        
        assert len(result.reasoning) == 1000  # Truncated to limit


class TestIdeaExtractionResponseParsing:
    """Test idea extraction response parsing."""
    
    def test_parse_idea_extraction_response_valid(self):
        """Test parsing valid idea extraction response."""
        response = '''
        {
            "ideas": [
                {
                    "title": "AI-Powered Newsletter Tool",
                    "description": "Automated newsletter generation using AI",
                    "category": "SaaS",
                    "target_market": "Content creators",
                    "value_proposition": "Save time on content creation",
                    "complexity_score": 7,
                    "confidence": 0.9,
                    "source_context": "Mentioned at 5:30 in the video"
                },
                {
                    "title": "Mobile Expense Tracker",
                    "description": "Simple expense tracking app",
                    "category": "Mobile App",
                    "complexity_score": 4,
                    "confidence": 0.7
                }
            ],
            "summary": "Two promising business ideas found",
            "total_ideas": 2
        }
        '''
        
        result = ResponseParser.parse_idea_extraction_response(response)
        
        assert isinstance(result, list)
        assert len(result) == 2
        
        first_idea = result[0]
        assert isinstance(first_idea, ParsedIdea)
        assert first_idea.title == "AI-Powered Newsletter Tool"
        assert first_idea.description == "Automated newsletter generation using AI"
        assert first_idea.category == "SaaS"
        assert first_idea.complexity_score == 7
        assert first_idea.confidence == 0.9
    
    def test_parse_idea_extraction_response_field_validation(self):
        """Test field validation and length limits in idea extraction."""
        long_title = "A" * 150  # Longer than 100 char limit
        long_description = "B" * 600  # Longer than 500 char limit
        
        response = f'''
        {{
            "ideas": [
                {{
                    "title": "{long_title}",
                    "description": "{long_description}",
                    "category": "Really Long Category Name That Exceeds Limit",
                    "complexity_score": 15,
                    "confidence": 1.5
                }}
            ]
        }}
        '''
        
        result = ResponseParser.parse_idea_extraction_response(response)
        
        assert len(result) == 1
        idea = result[0]
        
        assert len(idea.title) == 100  # Truncated
        assert len(idea.description) == 500  # Truncated
        assert len(idea.category) == 50  # Truncated
        assert idea.complexity_score == 10  # Clamped to max
        assert idea.confidence == 1.0  # Clamped to max
    
    def test_parse_idea_extraction_response_skip_invalid_ideas(self):
        """Test skipping ideas without required title/description."""
        response = '''
        {
            "ideas": [
                {
                    "title": "Valid Idea",
                    "description": "Valid description",
                    "category": "SaaS"
                },
                {
                    "title": "",
                    "description": "No title idea",
                    "category": "Mobile App"
                },
                {
                    "title": "No description idea",
                    "description": "",
                    "category": "E-commerce"
                },
                {
                    "description": "Missing title entirely",
                    "category": "Service"
                }
            ]
        }
        '''
        
        result = ResponseParser.parse_idea_extraction_response(response)
        
        assert len(result) == 1  # Only the valid idea
        assert result[0].title == "Valid Idea"
    
    def test_parse_idea_extraction_response_empty_ideas_list(self):
        """Test handling empty ideas list."""
        response = '{"ideas": []}'
        
        result = ResponseParser.parse_idea_extraction_response(response)
        
        assert result == []
    
    @patch('tubesensei.app.ai.response_parser.logger')
    def test_parse_idea_extraction_response_malformed_idea_logging(self, mock_logger):
        """Test logging when individual idea parsing fails."""
        response = '''
        {
            "ideas": [
                {
                    "title": "Good Idea",
                    "description": "Good description", 
                    "category": "SaaS"
                },
                {
                    "title": "Bad Idea",
                    "complexity_score": "not_a_number"
                }
            ]
        }
        '''
        
        result = ResponseParser.parse_idea_extraction_response(response)
        
        # Should have logged a warning for the malformed idea
        mock_logger.warning.assert_called()
        assert len(result) == 1  # Only the good idea remains


class TestQualityAssessmentResponseParsing:
    """Test quality assessment response parsing."""
    
    def test_parse_quality_assessment_response_valid(self):
        """Test parsing valid quality assessment response."""
        response = '''
        {
            "quality_score": 0.85,
            "viability_assessment": {
                "clarity": 0.9,
                "market_need": 0.8,
                "feasibility": 0.7,
                "uniqueness": 0.6
            },
            "strengths": ["Clear value proposition", "Large market"],
            "weaknesses": ["High competition", "Technical complexity"],
            "recommendations": ["Conduct market research", "Build MVP"]
        }
        '''
        
        result = ResponseParser.parse_quality_assessment_response(response)
        
        assert isinstance(result, ParsedQualityAssessment)
        assert result.quality_score == 0.85
        assert len(result.viability_scores) == 4
        assert result.viability_scores["clarity"] == 0.9
        assert len(result.strengths) == 2
        assert len(result.weaknesses) == 2
        assert len(result.recommendations) == 2
    
    def test_parse_quality_assessment_response_score_validation(self):
        """Test score validation and range clamping."""
        response = '''
        {
            "quality_score": 1.5,
            "viability_assessment": {
                "clarity": -0.5,
                "market_need": 2.0,
                "feasibility": "invalid"
            }
        }
        '''
        
        result = ResponseParser.parse_quality_assessment_response(response)
        
        assert result.quality_score == 1.0  # Clamped to max
        assert result.viability_scores["clarity"] == 0.0  # Clamped to min
        assert result.viability_scores["market_need"] == 1.0  # Clamped to max
        assert result.viability_scores["feasibility"] == 0.5  # Invalid becomes default
    
    def test_parse_quality_assessment_response_list_validation(self):
        """Test validation and length limits for list fields."""
        # Create lists longer than the 10-item limit
        long_strengths = [f"Strength {i}" for i in range(15)]
        long_weaknesses = [f"Weakness {i}" for i in range(12)]
        
        response = {
            "quality_score": 0.7,
            "strengths": long_strengths,
            "weaknesses": long_weaknesses,
            "recommendations": "not_a_list"  # Invalid type
        }
        
        result = ResponseParser.parse_quality_assessment_response(
            json.dumps(response)
        )
        
        assert len(result.strengths) == 10  # Truncated to limit
        assert len(result.weaknesses) == 10  # Truncated to limit
        assert result.recommendations == []  # Invalid becomes empty list
    
    def test_parse_quality_assessment_response_string_length_limits(self):
        """Test string length limits in list items."""
        long_string = "A" * 300  # Longer than 200 char limit for strengths
        
        response = f'''
        {{
            "quality_score": 0.8,
            "strengths": ["{long_string}"],
            "weaknesses": ["{long_string}"],
            "recommendations": ["{long_string}"]
        }}
        '''
        
        result = ResponseParser.parse_quality_assessment_response(response)
        
        assert len(result.strengths[0]) == 200  # Truncated
        assert len(result.weaknesses[0]) == 200  # Truncated 
        assert len(result.recommendations[0]) == 300  # Different limit for recommendations


class TestStructuredDataExtraction:
    """Test structured data extraction using regex patterns."""
    
    def test_extract_structured_data_basic(self):
        """Test basic regex pattern extraction."""
        text = """
        Name: John Doe
        Email: john@example.com
        Score: 85
        """
        
        patterns = {
            "name": r"Name:\s*(.+)",
            "email": r"Email:\s*(.+)",
            "score": r"Score:\s*(\d+)"
        }
        
        result = ResponseParser.extract_structured_data(text, patterns)
        
        assert result["name"] == "John Doe"
        assert result["email"] == "john@example.com"
        assert result["score"] == "85"
    
    def test_extract_structured_data_no_capturing_groups(self):
        """Test pattern extraction without capturing groups."""
        text = "The answer is 42 and the question is unknown."
        
        patterns = {
            "number": r"\d+",
            "question": r"question is \w+"
        }
        
        result = ResponseParser.extract_structured_data(text, patterns)
        
        assert result["number"] == "42"
        assert result["question"] == "question is unknown"
    
    def test_extract_structured_data_no_match(self):
        """Test behavior when pattern doesn't match."""
        text = "This text has no matches."
        
        patterns = {
            "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
            "phone": r"\d{3}-\d{3}-\d{4}"
        }
        
        result = ResponseParser.extract_structured_data(text, patterns)
        
        assert result["email"] is None
        assert result["phone"] is None
    
    @patch('tubesensei.app.ai.response_parser.logger')
    def test_extract_structured_data_invalid_regex(self, mock_logger):
        """Test handling of invalid regex patterns."""
        text = "Some text"
        
        patterns = {
            "valid": r"\w+",
            "invalid": r"[invalid regex pattern"  # Missing closing bracket
        }
        
        result = ResponseParser.extract_structured_data(text, patterns)
        
        assert result["valid"] == "Some"
        assert result["invalid"] is None
        mock_logger.warning.assert_called()
    
    def test_extract_structured_data_multiline_dotall(self):
        """Test regex with multiline and dotall flags."""
        text = """
        START
        Line 1
        Line 2
        END
        """
        
        patterns = {
            "content": r"START(.*?)END"
        }
        
        result = ResponseParser.extract_structured_data(text, patterns)
        
        # Should capture everything between START and END including newlines
        assert "Line 1" in result["content"]
        assert "Line 2" in result["content"]


class TestRangeValidation:
    """Test range validation in various parsing methods."""
    
    def test_confidence_score_range_0_to_1(self):
        """Test that confidence scores are properly clamped to 0.0-1.0 range."""
        test_cases = [
            (-0.5, 0.0),
            (0.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),
            (1.5, 1.0),
            (100, 1.0)
        ]
        
        for input_val, expected in test_cases:
            response = f'{{"is_valuable": true, "confidence_score": {input_val}}}'
            result = ResponseParser.parse_video_filter_response(response)
            assert result.confidence_score == expected
    
    def test_complexity_score_range_1_to_10(self):
        """Test that complexity scores are properly clamped to 1-10 range."""
        test_cases = [
            (-5, 1),
            (0, 1), 
            (1, 1),
            (5, 5),
            (10, 10),
            (15, 10),
            (100, 10)
        ]
        
        for input_val, expected in test_cases:
            response = f'''
            {{
                "ideas": [{{
                    "title": "Test",
                    "description": "Test desc",
                    "complexity_score": {input_val}
                }}]
            }}
            '''
            result = ResponseParser.parse_idea_extraction_response(response)
            assert len(result) == 1
            assert result[0].complexity_score == expected