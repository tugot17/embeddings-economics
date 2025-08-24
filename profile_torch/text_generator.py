import random
from typing import List, Tuple, Dict, Any


class TextGenerator:
    """Generate random text of varying lengths for embedding benchmarks."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
        # Sample vocabulary for generating random text
        self.words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "and", "runs", "through", "forest", "green", "mountain", "river",
            "flows", "gently", "under", "bright", "sun", "shines", "down",
            "upon", "beautiful", "landscape", "where", "birds", "sing",
            "songs", "of", "joy", "happiness", "peace", "tranquility",
            "nature", "provides", "sanctuary", "for", "all", "living",
            "creatures", "who", "seek", "refuge", "from", "busy", "world",
            "technology", "advances", "rapidly", "changing", "how", "we",
            "live", "work", "communicate", "with", "each", "other",
            "artificial", "intelligence", "machine", "learning", "deep",
            "neural", "networks", "computer", "science", "data", "analysis",
            "algorithms", "processing", "information", "knowledge", "wisdom",
            "understanding", "human", "experience", "consciousness",
            "philosophy", "science", "mathematics", "physics", "chemistry",
            "biology", "history", "culture", "society", "economics",
            "politics", "government", "education"
        ]
        
        self.sentence_starters = [
            "The researchers discovered that",
            "According to recent studies,",
            "In the field of artificial intelligence,",
            "Scientists have observed that",
            "The latest developments in technology show",
            "Experts believe that",
            "Recent experiments demonstrate",
            "Analysis of the data reveals",
            "The study concluded that",
            "Observations indicate that"
        ]
    
    def generate_text(self, target_length) -> str:
        """Generate random text of specified target length.
        
        Args:
            target_length: Either a string category ('short', 'medium', 'long')
                          or an integer specifying exact number of words
        """
        if isinstance(target_length, int):
            return self._generate_exact_words(target_length)
        elif target_length == "short":
            return self._generate_short_text()
        elif target_length == "medium":
            return self._generate_medium_text()
        elif target_length == "long":
            return self._generate_long_text()
        else:
            raise ValueError(f"Unknown target length: {target_length}")
    
    def _generate_exact_words(self, num_words: int) -> str:
        """Generate text with exactly the specified number of words."""
        if num_words <= 0:
            return ""
        
        words = random.choices(self.words, k=num_words)
        text = " ".join(words).capitalize()
        
        # Add appropriate punctuation based on length
        if num_words <= 20:
            return text + "."
        elif num_words <= 50:
            # Add a sentence starter for medium length
            starter = random.choice(self.sentence_starters)
            remaining_words = max(0, num_words - len(starter.split()))
            if remaining_words > 0:
                additional_words = random.choices(self.words, k=remaining_words)
                return starter + " " + " ".join(additional_words) + "."
            else:
                return starter + "."
        else:
            # For longer texts, create multiple sentences
            sentences = []
            words_remaining = num_words
            
            while words_remaining > 0:
                if words_remaining <= 15:
                    # Final short sentence
                    sentence_words = random.choices(self.words, k=words_remaining)
                    sentences.append(" ".join(sentence_words).capitalize() + ".")
                    words_remaining = 0
                else:
                    # Create a sentence with 10-25 words
                    sentence_length = min(random.randint(10, 25), words_remaining)
                    if len(sentences) == 0:
                        # First sentence gets a starter
                        starter = random.choice(self.sentence_starters)
                        starter_words = len(starter.split())
                        remaining = max(0, sentence_length - starter_words)
                        if remaining > 0:
                            additional = random.choices(self.words, k=remaining)
                            sentence = starter + " " + " ".join(additional) + "."
                        else:
                            sentence = starter + "."
                        words_remaining -= sentence_length
                    else:
                        # Subsequent sentences
                        sentence_words = random.choices(self.words, k=sentence_length)
                        sentence = " ".join(sentence_words).capitalize() + "."
                        words_remaining -= sentence_length
                    
                    sentences.append(sentence)
            
            return " ".join(sentences)

    def _generate_short_text(self) -> str:
        """Generate 5-15 word phrases."""
        num_words = random.randint(5, 15)
        words = random.choices(self.words, k=num_words)
        return " ".join(words).capitalize() + "."
    
    def _generate_medium_text(self) -> str:
        """Generate 20-50 word sentences."""
        starter = random.choice(self.sentence_starters)
        num_words = random.randint(15, 40)
        words = random.choices(self.words, k=num_words)
        return starter + " " + " ".join(words) + "."
    
    def _generate_long_text(self) -> str:
        """Generate 100-300 word paragraphs."""
        num_sentences = random.randint(4, 8)
        sentences = []
        
        for _ in range(num_sentences):
            starter = random.choice(self.sentence_starters)
            num_words = random.randint(15, 40)
            words = random.choices(self.words, k=num_words)
            sentence = starter + " " + " ".join(words) + "."
            sentences.append(sentence)
        
        return " ".join(sentences)
    
    def generate_batch(self, batch_size: int, length_distribution="mixed") -> List[str]:
        """Generate a batch of texts with specified length distribution.
        
        Args:
            batch_size: Number of texts to generate
            length_distribution: Either:
                - String category: 'short', 'medium', 'long', 'mixed'  
                - Integer: exact number of words for all texts
                - List of integers: specific word counts for mixed lengths
        """
        texts = []
        
        if isinstance(length_distribution, int):
            # All texts with the same exact word count
            for _ in range(batch_size):
                texts.append(self.generate_text(length_distribution))
        elif isinstance(length_distribution, list):
            # Mix of specific word counts
            for _ in range(batch_size):
                word_count = random.choice(length_distribution)
                texts.append(self.generate_text(word_count))
        elif length_distribution == "mixed":
            # Mix of different lengths
            lengths = ["short", "medium", "long"]
            for _ in range(batch_size):
                length = random.choice(lengths)
                texts.append(self.generate_text(length))
        elif length_distribution in ["short", "medium", "long"]:
            # All texts of the same category
            for _ in range(batch_size):
                texts.append(self.generate_text(length_distribution))
        else:
            raise ValueError(f"Unknown length distribution: {length_distribution}")
        
        return texts