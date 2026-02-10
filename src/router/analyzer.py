"""
Request analyzer for complexity classification and token counting.
Analyzes incoming requests to determine routing decisions.
"""

from dataclasses import dataclass
from typing import List, Optional
import re

from src.providers.base import LLMRequest, Message, ComplexityLevel
from src.utils.token_counter import TokenCounter


@dataclass
class RequestAnalysis:
    """Analysis result for an LLM request."""
    token_count: int
    estimated_output_tokens: int
    complexity: ComplexityLevel
    has_code: bool
    has_math: bool
    language_count: int
    question_type: str  # "simple", "analytical", "creative", "coding"
    estimated_cost_simple: float
    estimated_cost_complex: float
    recommended_model: str
    recommended_provider: str


class RequestAnalyzer:
    """Analyzes LLM requests to determine complexity and routing."""

    # Token thresholds for complexity classification
    SIMPLE_THRESHOLD = 200
    MODERATE_THRESHOLD = 1000

    # Patterns for content analysis
    CODE_PATTERNS = [
        r'```[\s\S]*?```',  # Code blocks
        r'def\s+\w+\s*\(',  # Python function
        r'function\s+\w+\s*\(',  # JavaScript function
        r'class\s+\w+',  # Class definition
        r'import\s+\w+',  # Import statements
        r'\w+\s*=\s*\[.*\]',  # Array assignment
        r'for\s+\w+\s+in',  # For loop
        r'if\s+\w+\s*[=<>!]',  # If statement
    ]

    MATH_PATTERNS = [
        r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Basic math
        r'∫|∑|∏|√|∂|∇',  # Math symbols
        r'equation|formula|calculate|compute',  # Math keywords
        r'\$.*?\$',  # LaTeX inline
        r'\\frac|\\sqrt|\\sum',  # LaTeX commands
    ]

    COMPLEX_KEYWORDS = [
        'analyze', 'compare', 'evaluate', 'synthesize', 'critique',
        'design', 'architect', 'implement', 'optimize', 'debug',
        'explain in detail', 'step by step', 'comprehensive',
        'research', 'investigate', 'deep dive'
    ]

    CODING_KEYWORDS = [
        'write a function', 'write a program', 'write code', 'code to',
        'implement a', 'create a function', 'create a class', 'create a script',
        'write a script', 'algorithm for', 'data structure',
        'python', 'javascript', 'java ', 'c++', 'typescript', 'rust ',
        'golang', 'sql query', 'regex', 'api endpoint', 'unit test',
        'refactor', 'sort', 'binary search', 'linked list', 'hash map',
        'leetcode', 'hackerrank', 'coding challenge',
    ]

    SIMPLE_KEYWORDS = [
        'what is', 'who is', 'when did', 'where is', 'define',
        'list', 'name', 'how many', 'yes or no', 'true or false',
        'translate', 'convert', 'summarize briefly'
    ]

    def __init__(self):
        self.token_counter = TokenCounter()

    def analyze(self, request: LLMRequest) -> RequestAnalysis:
        """
        Analyze a request to determine its complexity and routing.

        Args:
            request: LLM request to analyze

        Returns:
            RequestAnalysis with classification results
        """
        # Get prompt text
        prompt_text = request.prompt_text
        last_message = request.last_user_message or ""

        # Count tokens
        token_count = TokenCounter.count_message_tokens(request.messages)

        # Estimate output tokens (rough heuristic)
        estimated_output = self._estimate_output_tokens(token_count, prompt_text)

        # Analyze content
        has_code = self._has_code(prompt_text)
        has_math = self._has_math(prompt_text)
        language_count = self._count_languages(prompt_text)

        # Determine question type
        question_type = self._classify_question_type(last_message)

        # Classify complexity
        complexity = self._classify_complexity(
            token_count=token_count,
            has_code=has_code,
            has_math=has_math,
            question_type=question_type,
            prompt_text=prompt_text
        )

        # Get recommendations
        recommended_model, recommended_provider = self._get_recommendations(complexity)

        # Estimate costs
        cost_simple = self._estimate_cost("llama-3.1-8b-instant", token_count, estimated_output)
        cost_complex = self._estimate_cost("llama-3.3-70b-versatile", token_count, estimated_output)

        return RequestAnalysis(
            token_count=token_count,
            estimated_output_tokens=estimated_output,
            complexity=complexity,
            has_code=has_code,
            has_math=has_math,
            language_count=language_count,
            question_type=question_type,
            estimated_cost_simple=cost_simple,
            estimated_cost_complex=cost_complex,
            recommended_model=recommended_model,
            recommended_provider=recommended_provider,
        )

    def _has_code(self, text: str) -> bool:
        """Check if text contains code."""
        for pattern in self.CODE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _has_math(self, text: str) -> bool:
        """Check if text contains mathematical content."""
        for pattern in self.MATH_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _count_languages(self, text: str) -> int:
        """Estimate number of languages/domains in text."""
        # Simple heuristic based on content variety
        indicators = 0
        if self._has_code(text):
            indicators += 1
        if self._has_math(text):
            indicators += 1
        if re.search(r'[^\x00-\x7F]', text):  # Non-ASCII (likely other languages)
            indicators += 1
        return max(1, indicators)

    def _classify_question_type(self, text: str) -> str:
        """Classify the type of question/task."""
        text_lower = text.lower()

        # Check for coding/code-generation tasks first (highest priority)
        for keyword in self.CODING_KEYWORDS:
            if keyword in text_lower:
                return "coding"

        # Check for code patterns (code blocks, function defs, imports)
        if self._has_code(text) or 'code' in text_lower:
            return "coding"

        # Check for simple questions
        for keyword in self.SIMPLE_KEYWORDS:
            if keyword in text_lower:
                return "simple"

        # Check for complex/analytical questions
        for keyword in self.COMPLEX_KEYWORDS:
            if keyword in text_lower:
                return "analytical"

        # Check for creative tasks
        creative_keywords = ['write', 'create', 'generate', 'compose', 'story', 'poem']
        for keyword in creative_keywords:
            if keyword in text_lower:
                return "creative"

        return "simple"

    def _classify_complexity(
        self,
        token_count: int,
        has_code: bool,
        has_math: bool,
        question_type: str,
        prompt_text: str
    ) -> ComplexityLevel:
        """
        Classify request complexity based on multiple factors.

        Complexity is determined by:
        1. Token count (primary factor)
        2. Content type (code, math)
        3. Question type
        4. Presence of complex keywords
        """
        # Base classification from token count
        if token_count < self.SIMPLE_THRESHOLD:
            base_complexity = ComplexityLevel.SIMPLE
        elif token_count < self.MODERATE_THRESHOLD:
            base_complexity = ComplexityLevel.MODERATE
        else:
            base_complexity = ComplexityLevel.COMPLEX

        # Adjust based on content
        complexity_score = 0

        if base_complexity == ComplexityLevel.SIMPLE:
            complexity_score = 1
        elif base_complexity == ComplexityLevel.MODERATE:
            complexity_score = 2
        else:
            complexity_score = 3

        # Bump up complexity for code/math
        if has_code:
            complexity_score += 1
        if has_math:
            complexity_score += 1

        # Adjust for question type
        if question_type == "coding":
            complexity_score += 1.5
        elif question_type == "analytical":
            complexity_score += 1
        elif question_type == "technical":
            complexity_score += 1

        # Check for complex keywords
        text_lower = prompt_text.lower()
        for keyword in self.COMPLEX_KEYWORDS:
            if keyword in text_lower:
                complexity_score += 0.5

        # Final classification
        if complexity_score <= 1.5:
            return ComplexityLevel.SIMPLE
        elif complexity_score <= 3:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.COMPLEX

    def _estimate_output_tokens(self, input_tokens: int, prompt_text: str) -> int:
        """Estimate expected output tokens."""
        # Base ratio
        ratio = 1.5

        # Adjust for question type
        text_lower = prompt_text.lower()

        if 'summarize' in text_lower or 'briefly' in text_lower:
            ratio = 0.5
        elif 'explain in detail' in text_lower or 'comprehensive' in text_lower:
            ratio = 2.5
        elif 'list' in text_lower:
            ratio = 1.0
        elif 'write' in text_lower or 'create' in text_lower:
            ratio = 2.0

        estimated = int(input_tokens * ratio)

        # Cap at reasonable limits
        return min(max(estimated, 50), 4000)

    def _get_recommendations(self, complexity: ComplexityLevel) -> tuple:
        """Get recommended model and provider based on complexity."""
        recommendations = {
            ComplexityLevel.SIMPLE: ("llama-3.1-8b-instant", "groq"),
            ComplexityLevel.MODERATE: ("mixtral-8x7b-32768", "groq"),
            ComplexityLevel.COMPLEX: ("llama-3.3-70b-versatile", "groq"),
        }
        return recommendations.get(complexity, ("llama-3.1-8b-instant", "groq"))

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a model."""
        # Groq pricing per 1K tokens
        pricing = {
            "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
            "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
            "mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024},
        }

        if model not in pricing:
            return 0.0

        prices = pricing[model]
        input_cost = (input_tokens / 1000) * prices["input"]
        output_cost = (output_tokens / 1000) * prices["output"]

        return round(input_cost + output_cost, 6)


# Global analyzer instance
_analyzer: Optional[RequestAnalyzer] = None


def get_analyzer() -> RequestAnalyzer:
    """Get the global request analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = RequestAnalyzer()
    return _analyzer


def analyze_request(request: LLMRequest) -> RequestAnalysis:
    """Convenience function to analyze a request."""
    return get_analyzer().analyze(request)
