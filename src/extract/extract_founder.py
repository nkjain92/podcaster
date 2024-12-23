import os
from typing import List, Dict, Any, Tuple
import json
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import tiktoken
from datetime import datetime
import asyncio

from .founder_types import *

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=api_key)
encoding = tiktoken.get_encoding("cl100k_base")

# Initialize async client
async_client = AsyncOpenAI(api_key=api_key)

def count_tokens(text: str) -> int:
    """Count tokens in text"""
    return len(encoding.encode(text))

def chunk_transcript(transcript: str, max_tokens: int = 100000) -> List[str]:
    """Split transcript into chunks that fit within token limit"""
    chunks = []
    current_chunk = []
    current_tokens = 0

    # Split by paragraphs
    paragraphs = transcript.split("\n\n")

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def extract_with_retries(prompt: str, max_retries: int = 3) -> Dict:
    """Make API call with retries and error handling"""
    start_time = datetime.now()

    for attempt in range(max_retries):
        try:
            print(f"Making API call (attempt {attempt + 1}/{max_retries})...")
            response = client.chat.completions.create(
                model="o1-preview-2024-09-12",
                messages=[{
                    "role": "user",
                    "content": "You must respond with pure JSON only, no markdown, no explanations, no code blocks.\n\n" + prompt
                }]
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"API call successful (took {elapsed:.1f}s)")

            # Check if response is not empty
            content = response.choices[0].message.content.strip()
            if not content:
                raise ValueError("Empty response from API")

            print(f"Raw response content: {content}")

            # Remove any potential markdown or code block formatting
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:-1])

            return json.loads(content)
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"API call failed after {elapsed:.1f}s: {e}")
            if attempt == max_retries - 1:
                raise e
            continue

async def extract_with_retries_async(prompt: str, extraction_type: str, max_retries: int = 3) -> Tuple[str, Dict]:
    """Async version of extract_with_retries"""
    start_time = datetime.now()

    for attempt in range(max_retries):
        try:
            print(f"Making API call for {extraction_type} (attempt {attempt + 1}/{max_retries})...")
            response = await async_client.chat.completions.create(
                model="o1-preview-2024-09-12",
                messages=[{
                    "role": "user",
                    "content": "You must respond with pure JSON only, no markdown, no explanations, no code blocks.\n\n" + prompt
                }]
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"API call for {extraction_type} successful (took {elapsed:.1f}s)")

            # Check if response is not empty
            content = response.choices[0].message.content.strip()
            if not content:
                raise ValueError("Empty response from API")

            print(f"Raw response content for {extraction_type}: {content}")

            # Remove any potential markdown or code block formatting
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:-1])

            return extraction_type, json.loads(content)
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"API call for {extraction_type} failed after {elapsed:.1f}s: {e}")
            if attempt == max_retries - 1:
                raise e
            continue

async def extract_all_components(transcript: str) -> Dict[str, Any]:
    """Extract all components in parallel"""
    tasks = [
        extract_with_retries_async(extract_traits.__doc__, "traits"),
        extract_with_retries_async(extract_beliefs.__doc__, "beliefs"),
        extract_with_retries_async(extract_philosophies.__doc__, "philosophies"),
        extract_with_retries_async(extract_failures.__doc__, "failures"),
        extract_with_retries_async(extract_key_decisions.__doc__, "key_decisions"),
        extract_with_retries_async(extract_mentors_and_network.__doc__, "mentors"),
        extract_with_retries_async(extract_habits.__doc__, "habits"),
        extract_with_retries_async(extract_unique_approaches.__doc__, "approaches"),
        extract_with_retries_async(extract_anecdotes.__doc__, "anecdotes"),
        extract_with_retries_async(extract_emotional_intelligence.__doc__, "emotional_intelligence")
    ]

    results = {}
    for task in asyncio.as_completed(tasks):
        extraction_type, data = await task
        results[extraction_type] = data

    return results

def extract_basic_info(transcript: str) -> BasicInfo:
    """Extract basic info about founder from transcript"""
    prompt = f"""You must respond with ONLY a JSON object, no markdown formatting, no explanations, no code blocks.
    The JSON must exactly match this schema:
    {{
      "name": "string",
      "domain": ["string", ...],
      "financial_background": "string or null",
      "era": {{
        "start": "YYYY",
        "end": "YYYY or null"
      }}
    }}

    Analyze this transcript and extract basic information about the founder:
    {transcript[:2000]}...
    """

    result = extract_with_retries(prompt)

    return BasicInfo(
        name=result["name"],
        domain=result["domain"],
        financial_background=result.get("financial_background"),
        era=Era(**result["era"]) if "era" in result else None
    )

def extract_timeline(transcript: str) -> List[TimelineEvent]:
    """Extract timeline of key events"""
    prompt = f"""Analyze this transcript and extract key events in chronological order.
    Return a JSON object in this exact format:
    {{
      "events": [
        {{
          "date": "YYYY or YYYY-MM-DD",
          "event_type": "FOUNDING|PIVOT|CRISIS|SUCCESS|OTHER",
          "description": "string",
          "emotional_context": "string or null",
          "trait_changes": [
            {{
              "trait": "string",
              "change": "string"
            }}
          ],
          "alternative_paths": ["string"],
          "external_triggers": ["string"],
          "source": "quote from transcript",
          "confidence": number between 0 and 1,
          "data_type": "FACTUAL|SPECULATIVE|INFERRED"
        }}
      ]
    }}

    Example response:
    {{
      "events": [
        {{
          "date": "1995-06",
          "event_type": "FOUNDING",
          "description": "Founded Zip2, his first internet company",
          "emotional_context": "Determined and ambitious, working long hours",
          "trait_changes": [
            {{
              "trait": "Work ethic",
              "change": "Developed intense work habits"
            }}
          ],
          "alternative_paths": ["Considered pursuing PhD at Stanford"],
          "external_triggers": ["Rise of internet companies in Silicon Valley"],
          "source": "As mentioned in the transcript: 'Musk and his brother started Zip2 in 1995'",
          "confidence": 0.9,
          "data_type": "FACTUAL"
        }}
      ]
    }}

    - Return empty array if no events found
    - Include exact quotes as sources where possible
    - Focus on major events that shaped the founder's journey
    - Ensure trait_changes is always an array of objects with trait and change fields

    Transcript: {transcript}
    """

    result = extract_with_retries(prompt)
    events = []

    for event_data in result.get("events", []):
        try:
            trait_changes = tuple(TraitChange(**tc) for tc in event_data.get("trait_changes", []))
            alternative_paths = tuple(event_data.get("alternative_paths", []))
            external_triggers = tuple(event_data.get("external_triggers", []))

            events.append(TimelineEvent(
                date=event_data["date"],
                event_type=EventType[event_data["event_type"]],
                description=event_data["description"],
                emotional_context=event_data.get("emotional_context"),
                trait_changes=trait_changes,
                alternative_paths=alternative_paths,
                external_triggers=external_triggers,
                source=event_data["source"],
                confidence=event_data["confidence"],
                data_type=DataType[event_data["data_type"]]
            ))
        except Exception as e:
            print(f"Warning: Failed to parse event: {e}")
            continue

    return events

def extract_traits(transcript: str) -> List[Trait]:
    """Extract personality traits and skills"""
    prompt = f"""Analyze this transcript and identify key personality traits and skills.
    Return a JSON object in this exact format:
    {{
      "traits": [
        {{
          "trait": "string",
          "category": "TRAIT|SKILL",
          "origin": "INNATE|DEVELOPED",
          "description": "string",
          "examples": [
            {{
              "description": "string",
              "source": "string"
            }}
          ],
          "evolution": [
            {{
              "period": "string",
              "change": "string",
              "trigger": "string or null"
            }}
          ],
          "blind_spots": [
            {{
              "description": "string",
              "overcome_strategy": "string or null",
              "source": "string",
              "confidence": number between 0 and 1,
              "data_type": "FACTUAL|SPECULATIVE|INFERRED"
            }}
          ],
          "source": "string",
          "confidence": number between 0 and 1,
          "data_type": "FACTUAL|SPECULATIVE|INFERRED"
        }}
      ]
    }}
    """

    result = extract_with_retries(prompt)
    traits = []

    for trait_data in result["traits"]:
        traits.append(Trait(
            trait=trait_data["trait"],
            category=TraitCategory[trait_data["category"]],
            origin=TraitOrigin[trait_data["origin"]],
            description=trait_data["description"],
            examples=[Example(**ex) for ex in trait_data.get("examples", [])],
            evolution=[TraitEvolution(**ev) for ev in trait_data.get("evolution", [])],
            blind_spots=[BlindSpot(**bs) for bs in trait_data.get("blind_spots", [])],
            source=trait_data["source"],
            confidence=trait_data["confidence"],
            data_type=DataType[trait_data["data_type"]]
        ))

    return traits

def extract_beliefs(transcript: str) -> List[Belief]:
    """Extract beliefs and rationales"""
    prompt = f"""Analyze this transcript and identify the founder's core beliefs and their rationales.
    Return a JSON object in this exact format:
    {{
      "beliefs": [
        {{
          "belief": "string",
          "rationale": "string or null",
          "source": "quote from transcript",
          "confidence": number between 0 and 1,
          "data_type": "FACTUAL|SPECULATIVE|INFERRED"
        }}
      ]
    }}

    Example response:
    {{
      "beliefs": [
        {{
          "belief": "Innovation requires taking big risks",
          "rationale": "Consistently demonstrated through major business decisions",
          "source": "As stated in transcript: 'You have to be willing to take risks to create something truly new'",
          "confidence": 0.9,
          "data_type": "FACTUAL"
        }}
      ]
    }}

    - Return empty array if no beliefs found
    - Include exact quotes as sources where possible

    Transcript: {transcript}
    """

    result = extract_with_retries(prompt)
    beliefs = []

    for belief_data in result.get("beliefs", []):
        try:
            beliefs.append(Belief(
                belief=belief_data["belief"],
                rationale=belief_data.get("rationale"),
                source=belief_data["source"],
                confidence=belief_data["confidence"],
                data_type=DataType[belief_data["data_type"]]
            ))
        except Exception as e:
            print(f"Warning: Failed to parse belief: {e}")
            continue

    return beliefs

def extract_philosophies(transcript: str) -> List[Philosophy]:
    """Extract philosophies and principles"""
    prompt = f"""Analyze this transcript and identify the founder's philosophies and principles.
    Return a JSON object in this exact format:
    {{
      "philosophies": [
        {{
          "principle": "string",
          "quotes": [
            {{
              "quote": "exact quote from transcript",
              "context": "string or null",
              "source": "where in transcript"
            }}
          ],
          "contradictions": [
            {{
              "description": "string",
              "timeline_ref": "string or null"
            }}
          ],
          "source": "quote from transcript",
          "confidence": number between 0 and 1,
          "data_type": "FACTUAL|SPECULATIVE|INFERRED"
        }}
      ]
    }}

    Example response:
    {{
      "philosophies": [
        {{
          "principle": "First principles thinking",
          "quotes": [
            {{
              "quote": "The best way to solve problems is to break them down to their fundamental truths",
              "context": "Discussing his approach to innovation",
              "source": "From section about SpaceX founding"
            }}
          ],
          "contradictions": [
            {{
              "description": "Sometimes relies on intuition over systematic analysis",
              "timeline_ref": "2008 Tesla crisis"
            }}
          ],
          "source": "Multiple instances throughout transcript",
          "confidence": 0.9,
          "data_type": "FACTUAL"
        }}
      ]
    }}

    - Return empty array if no philosophies found
    - Include exact quotes where possible
    - Note any contradictions or evolution in thinking

    Transcript: {transcript}
    """

    result = extract_with_retries(prompt)
    philosophies = []

    for philosophy_data in result.get("philosophies", []):
        try:
            philosophies.append(Philosophy(
                principle=philosophy_data["principle"],
                quotes=[Quote(**quote) for quote in philosophy_data.get("quotes", [])],
                contradictions=[Contradiction(**contradiction) for contradiction in philosophy_data.get("contradictions", [])],
                source=philosophy_data["source"],
                confidence=philosophy_data["confidence"],
                data_type=DataType[philosophy_data["data_type"]]
            ))
        except Exception as e:
            print(f"Warning: Failed to parse philosophy: {e}")
            continue

    return philosophies

def extract_failures(transcript: str) -> List[Failure]:
    """Extract failures and lessons learned"""
    prompt = f"""Analyze this transcript and identify significant failures and setbacks experienced by the founder.
    Return a JSON object in this exact format:
    {{
      "failures": [
        {{
          "event": "string",
          "date": "YYYY or YYYY-MM",
          "description": "string",
          "lessons_learned": ["string", ...],
          "source": "quote from transcript",
          "confidence": number between 0 and 1,
          "data_type": "FACTUAL|SPECULATIVE|INFERRED"
        }}
      ]
    }}

    Example response:
    {{
      "failures": [
        {{
          "event": "SpaceX First Three Launch Failures",
          "date": "2006-2008",
          "description": "First three Falcon 1 rockets failed to reach orbit, nearly bankrupting the company",
          "lessons_learned": [
            "Persistence through setbacks is crucial",
            "Technical problems require iterative solutions",
            "Team morale management during crises is essential"
          ],
          "source": "As mentioned: 'The first three launches ended in failure, nearly bankrupting both SpaceX and Tesla'",
          "confidence": 0.95,
          "data_type": "FACTUAL"
        }},
        {{
          "event": "Tesla Model 3 Production Hell",
          "date": "2017",
          "description": "Severe production delays and automation issues with Model 3",
          "lessons_learned": [
            "Over-automation can be counterproductive",
            "Manufacturing requires balance of human and machine work"
          ],
          "source": "Quote: 'Excessive automation at Tesla was a mistake. Humans are underrated.'",
          "confidence": 0.9,
          "data_type": "FACTUAL"
        }}
      ]
    }}

    - Return empty array if no failures found
    - Include exact quotes as sources where possible
    - Focus on significant failures that led to learning or changes
    - Include specific lessons learned from each failure

    Transcript: {transcript}
    """

    result = extract_with_retries(prompt)
    failures = []

    for failure_data in result.get("failures", []):
        try:
            failures.append(Failure(
                event=failure_data["event"],
                date=failure_data["date"],
                description=failure_data["description"],
                lessons_learned=failure_data.get("lessons_learned", []),
                source=failure_data["source"],
                confidence=failure_data["confidence"],
                data_type=DataType[failure_data["data_type"]]
            ))
        except Exception as e:
            print(f"Warning: Failed to parse failure: {e}")
            continue

    return failures

def extract_key_decisions(transcript: str) -> List[KeyDecision]:
    """Extract key decisions and their impacts"""
    prompt = f"""Analyze this transcript and identify key decisions made by the founder and their impacts.
    Return a JSON object in this exact format:
    {{
      "key_decisions": [
        {{
          "decision": "string",
          "date": "YYYY or YYYY-MM",
          "context": "string",
          "impact": "string",
          "reasoning": "string or null",
          "source": "quote from transcript",
          "confidence": number between 0 and 1,
          "data_type": "FACTUAL|SPECULATIVE|INFERRED"
        }}
      ]
    }}

    Example response:
    {{
      "key_decisions": [
        {{
          "decision": "Starting SpaceX instead of buying Russian rockets",
          "date": "2002",
          "context": "After selling PayPal, sought to advance space exploration",
          "impact": "Created first private company to reach orbit and revolutionized space industry",
          "reasoning": "Calculated that rockets could be built for fraction of current prices",
          "source": "As mentioned: 'After the Russians refused to sell him a rocket, Musk decided to build his own'",
          "confidence": 0.95,
          "data_type": "FACTUAL"
        }},
        {{
          "decision": "Taking over as Tesla CEO",
          "date": "2008",
          "context": "Company was struggling with production and management issues",
          "impact": "Transformed Tesla into leading electric vehicle company",
          "reasoning": "Believed in the mission and saw potential despite challenges",
          "source": "Quote: 'The company needed new leadership and I had to step in'",
          "confidence": 0.9,
          "data_type": "FACTUAL"
        }}
      ]
    }}

    - Return empty array if no key decisions found
    - Include exact quotes as sources where possible
    - Focus on decisions that had significant impact
    - Include context and reasoning when available

    Transcript: {transcript}
    """

    result = extract_with_retries(prompt)
    decisions = []

    for decision_data in result.get("key_decisions", []):
        try:
            decisions.append(KeyDecision(
                decision=decision_data["decision"],
                date=decision_data["date"],
                context=decision_data["context"],
                impact=decision_data["impact"],
                reasoning=decision_data.get("reasoning"),
                source=decision_data["source"],
                confidence=decision_data["confidence"],
                data_type=DataType[decision_data["data_type"]]
            ))
        except Exception as e:
            print(f"Warning: Failed to parse decision: {e}")
            continue

    return decisions

def extract_mentors_and_network(transcript: str) -> List[Connection]:
    """Extract mentors and network connections"""
    prompt = f"""Analyze this transcript and identify the founder's mentors and key network connections.
    Return a JSON object in this exact format:
    {{
      "connections": [
        {{
          "name": "string",
          "relationship": "string",
          "impact": "string",
          "period": "string or null",
          "source": "quote from transcript",
          "confidence": number between 0 and 1,
          "data_type": "FACTUAL|SPECULATIVE|INFERRED"
        }}
      ]
    }}

    Example response:
    {{
      "connections": [
        {{
          "name": "Peter Thiel",
          "relationship": "Co-founder and mentor",
          "impact": "Helped shape PayPal's strategy and provided crucial early guidance",
          "period": "1999-2002",
          "source": "Quote: 'Peter Thiel became a key advisor and helped shape the company's direction'",
          "confidence": 0.9,
          "data_type": "FACTUAL"
        }},
        {{
          "name": "Steve Jurvetson",
          "relationship": "Early investor and advisor",
          "impact": "Provided critical funding and support during Tesla's early days",
          "period": "2004-2008",
          "source": "As mentioned: 'Jurvetson's firm was one of the first to believe in Tesla's vision'",
          "confidence": 0.85,
          "data_type": "FACTUAL"
        }}
      ]
    }}

    - Return empty array if no connections found
    - Include exact quotes as sources where possible
    - Focus on relationships that had significant impact
    - Include time periods when mentioned

    Transcript: {transcript}
    """

    result = extract_with_retries(prompt)
    connections = []

    for connection_data in result.get("connections", []):
        try:
            connections.append(Connection(
                name=connection_data["name"],
                relationship=connection_data["relationship"],
                impact=connection_data["impact"],
                period=connection_data.get("period"),
                source=connection_data["source"],
                confidence=connection_data["confidence"],
                data_type=DataType[connection_data["data_type"]]
            ))
        except Exception as e:
            print(f"Warning: Failed to parse connection: {e}")
            continue

    return connections

def extract_habits(transcript: str) -> List[Habit]:
    """Extract daily habits and routines"""
    prompt = f"""Analyze this transcript and identify the founder's daily habits and routines.
    Return a JSON object in this exact format:
    {{
      "habits": [
        {{
          "habit": "string",
          "description": "string",
          "impact": "string or null",
          "source": "quote from transcript",
          "confidence": number between 0 and 1,
          "data_type": "FACTUAL|SPECULATIVE|INFERRED"
        }}
      ]
    }}

    Example response:
    {{
      "habits": [
        {{
          "habit": "Early morning work schedule",
          "description": "Wakes up at 5am and immediately starts working",
          "impact": "Maximizes productivity and focus during quiet hours",
          "source": "Quote: 'I start my day at 5am because that's when I can think most clearly'",
          "confidence": 0.9,
          "data_type": "FACTUAL"
        }},
        {{
          "habit": "Reading technical papers",
          "description": "Spends evenings reading scientific and engineering papers",
          "impact": "Maintains deep technical knowledge across multiple fields",
          "source": "As mentioned: 'I spend several hours each night reading technical literature'",
          "confidence": 0.85,
          "data_type": "FACTUAL"
        }}
      ]
    }}

    - Return empty array if no habits found
    - Include exact quotes as sources where possible
    - Focus on regular habits and routines
    - Include impact when mentioned

    Transcript: {transcript}
    """

    result = extract_with_retries(prompt)
    habits = []

    for habit_data in result.get("habits", []):
        try:
            habits.append(Habit(
                habit=habit_data["habit"],
                description=habit_data["description"],
                impact=habit_data.get("impact"),
                source=habit_data["source"],
                confidence=habit_data["confidence"],
                data_type=DataType[habit_data["data_type"]]
            ))
        except Exception as e:
            print(f"Warning: Failed to parse habit: {e}")
            continue

    return habits

def extract_unique_approaches(transcript: str) -> List[UniqueApproach]:
    """Extract unique approaches and strategies"""
    prompt = f"""Analyze this transcript and identify any unique approaches or strategies used by the founder.
    Return a JSON object in this exact format:
    {{
      "approaches": [
        {{
          "approach_name": "string",
          "description": "string",
          "source": "quote from transcript",
          "confidence": number between 0 and 1,
          "data_type": "FACTUAL|SPECULATIVE|INFERRED"
        }}
      ]
    }}

    Example response:
    {{
      "approaches": [
        {{
          "approach_name": "First Principles Thinking",
          "description": "Breaking down complex problems to fundamental truths and rebuilding from there",
          "source": "Quote: 'Instead of accepting industry standards, Musk would break problems down to their atomic level'",
          "confidence": 0.95,
          "data_type": "FACTUAL"
        }},
        {{
          "approach_name": "Vertical Integration",
          "description": "Building most components in-house to control quality and reduce costs",
          "source": "As mentioned: 'SpaceX manufactures over 80% of their rocket components internally'",
          "confidence": 0.9,
          "data_type": "FACTUAL"
        }}
      ]
    }}

    - Return empty array if no unique approaches found
    - Include exact quotes as sources where possible
    - Focus on approaches that differentiate from industry norms
    - Include specific examples when available

    Transcript: {transcript}
    """

    result = extract_with_retries(prompt)
    approaches = []

    for approach_data in result.get("approaches", []):
        try:
            approaches.append(UniqueApproach(
                approach_name=approach_data["approach_name"],
                description=approach_data["description"],
                source=approach_data["source"],
                confidence=approach_data["confidence"],
                data_type=DataType[approach_data["data_type"]]
            ))
        except Exception as e:
            print(f"Warning: Failed to parse approach: {e}")
            continue

    return approaches

def extract_anecdotes(transcript: str) -> List[Anecdote]:
    """Extract anecdotes and stories"""
    prompt = f"""Analyze this transcript and identify meaningful anecdotes or stories about the founder.
    Return a JSON object in this exact format:
    {{
      "anecdotes": [
        {{
          "title": "string",
          "narrative": "string",
          "moral": "string or null",
          "timeline_ref": "string or null",
          "source": "quote from transcript",
          "confidence": number between 0 and 1,
          "data_type": "FACTUAL|SPECULATIVE|INFERRED"
        }}
      ]
    }}

    Example response:
    {{
      "anecdotes": [
        {{
          "title": "Sleeping on the Factory Floor",
          "narrative": "During Model 3 production crisis, Musk slept on the factory floor to solve problems in real-time",
          "moral": "Leadership requires personal sacrifice and hands-on involvement",
          "timeline_ref": "2018 Tesla Production Hell",
          "source": "Quote: 'I slept on the factory floor to show that I care more about their hardship than my own comfort'",
          "confidence": 0.9,
          "data_type": "FACTUAL"
        }},
        {{
          "title": "Russian Rocket Negotiation",
          "narrative": "After Russians quoted high prices for rockets, Musk decided to build his own",
          "moral": "Sometimes creating your own solution is better than accepting existing options",
          "timeline_ref": "2002 SpaceX Founding",
          "source": "As mentioned: 'The Russians were asking too much, so Musk opened Excel and started calculating rocket costs'",
          "confidence": 0.95,
          "data_type": "FACTUAL"
        }}
      ]
    }}

    - Return empty array if no anecdotes found
    - Include exact quotes as sources where possible
    - Focus on stories that illustrate character or decision-making
    - Include moral lessons when apparent

    Transcript: {transcript}
    """

    result = extract_with_retries(prompt)
    anecdotes = []

    for anecdote_data in result.get("anecdotes", []):
        try:
            anecdotes.append(Anecdote(
                title=anecdote_data["title"],
                narrative=anecdote_data["narrative"],
                moral=anecdote_data.get("moral"),
                timeline_ref=anecdote_data.get("timeline_ref"),
                source=anecdote_data["source"],
                confidence=anecdote_data["confidence"],
                data_type=DataType[anecdote_data["data_type"]]
            ))
        except Exception as e:
            print(f"Warning: Failed to parse anecdote: {e}")
            continue

    return anecdotes

def extract_emotional_intelligence(transcript: str) -> EmotionalIntelligence:
    """Extract emotional intelligence aspects"""
    prompt = f"""Analyze this transcript and identify aspects of the founder's emotional intelligence.
    Return a JSON object in this exact format:
    {{
      "emotional_intelligence": {{
        "stress_management": [
          {{
            "strategy": "string",
            "context": "string",
            "effectiveness": "string or null"
          }}
        ],
        "leadership_style": [
          {{
            "trait": "string",
            "description": "string",
            "examples": ["string", ...]
          }}
        ],
        "source": "quote from transcript",
        "confidence": number between 0 and 1,
        "data_type": "FACTUAL|SPECULATIVE|INFERRED"
      }}
    }}

    Example response:
    {{
      "emotional_intelligence": {{
        "stress_management": [
          {{
            "strategy": "Focused Problem-Solving",
            "context": "During company crises",
            "effectiveness": "Highly effective at maintaining clarity under pressure"
          }},
          {{
            "strategy": "Direct Communication",
            "context": "Team management",
            "effectiveness": "Helps maintain team alignment during stress"
          }}
        ],
        "leadership_style": [
          {{
            "trait": "Lead by Example",
            "description": "Works alongside team during crises",
            "examples": [
              "Sleeping on factory floor during production issues",
              "Taking on engineering problems personally"
            ]
          }}
        ],
        "source": "Multiple instances throughout transcript",
        "confidence": 0.85,
        "data_type": "FACTUAL"
      }}
    }}

    - Include specific examples and contexts
    - Focus on emotional management and leadership approaches
    - Include effectiveness when mentioned
    - Use exact quotes where possible

    Transcript: {transcript}
    """

    result = extract_with_retries(prompt)
    ei_data = result["emotional_intelligence"]

    return EmotionalIntelligence(
        stress_management=[StressStrategy(**strategy) for strategy in ei_data.get("stress_management", [])],
        leadership_style=[LeadershipTrait(**trait) for trait in ei_data.get("leadership_style", [])],
        source=ei_data["source"],
        confidence=ei_data["confidence"],
        data_type=DataType[ei_data["data_type"]]
    )

def extract_founder_data(transcript: str) -> Founder:
    """Extract all founder data from transcript"""
    results = {}  # Initialize results at the start
    try:
        print("\n=== Starting Founder Data Extraction ===")

        # Handle large transcripts
        print("Chunking transcript...")
        chunks = chunk_transcript(transcript)
        print(f"Split into {len(chunks)} chunks")

        # Extract basic info from first chunk (keep this sequential)
        print("\nExtracting basic info...")
        basic_info = extract_basic_info(chunks[0])
        print(f"Found basic info for: {basic_info.name}")

        # Extract timeline events from all chunks
        print("\nExtracting timeline events...")
        all_timeline_events = []
        for i, chunk in enumerate(chunks, 1):
            try:
                print(f"Processing chunk {i}/{len(chunks)}...")
                events = extract_timeline(chunk)
                all_timeline_events.extend(events)
                print(f"Found {len(events)} events in chunk {i}")
            except Exception as e:
                print(f"Warning: Error extracting timeline from chunk {i}: {e}")

        # Sort and deduplicate timeline events
        timeline = sorted(set(all_timeline_events), key=lambda x: x.date)

        # Extract all other components in parallel
        print("\nExtracting additional components in parallel...")
        results = asyncio.run(extract_all_components(transcript))

        # Debug logging
        print("\nProcessing results:")
        for key, value in results.items():
            print(f"{key}: {type(value)}")
            if isinstance(value, dict):
                print(f"Keys: {value.keys()}")

        # Process all results
        traits = [Trait(**t) for t in results.get("traits", {}).get("traits", [])]
        print(f"Processed {len(traits)} traits")

        beliefs = [Belief(**b) for b in results.get("beliefs", {}).get("beliefs", [])]
        print(f"Processed {len(beliefs)} beliefs")

        philosophies = [Philosophy(**p) for p in results.get("philosophies", {}).get("philosophies", [])]
        print(f"Processed {len(philosophies)} philosophies")

        failures = [Failure(**f) for f in results.get("failures", {}).get("failures", [])]
        print(f"Processed {len(failures)} failures")

        key_decisions = [KeyDecision(**k) for k in results.get("key_decisions", {}).get("key_decisions", [])]
        print(f"Processed {len(key_decisions)} key decisions")

        mentors = [Connection(**m) for m in results.get("mentors", {}).get("connections", [])]
        print(f"Processed {len(mentors)} mentors")

        habits = [Habit(**h) for h in results.get("habits", {}).get("habits", [])]
        print(f"Processed {len(habits)} habits")

        approaches = [UniqueApproach(**a) for a in results.get("approaches", {}).get("approaches", [])]
        print(f"Processed {len(approaches)} unique approaches")

        anecdotes = [Anecdote(**a) for a in results.get("anecdotes", {}).get("anecdotes", [])]
        print(f"Processed {len(anecdotes)} anecdotes")

        # Process emotional intelligence
        ei_data = results.get("emotional_intelligence", {}).get("emotional_intelligence", {})
        emotional_intelligence = EmotionalIntelligence(
            stress_management=[StressStrategy(**s) for s in ei_data.get("stress_management", [])],
            leadership_style=[LeadershipTrait(**l) for l in ei_data.get("leadership_style", [])],
            source=ei_data.get("source", ""),
            confidence=ei_data.get("confidence", 0.0),
            data_type=DataType[ei_data.get("data_type", "FACTUAL")]
        )

        return Founder(
            basic_info=basic_info,
            early_life=[],  # Not implemented yet
            timeline=timeline,
            traits=traits,
            beliefs=beliefs,
            philosophies=philosophies,
            failures=failures,
            key_decisions=key_decisions,
            mentors_and_network=mentors,
            habits=habits,
            unique_approaches=approaches,
            anecdotes=anecdotes,
            emotional_intelligence=emotional_intelligence,
            metadata=Metadata(
                last_updated=datetime.now(),
                version="0.1",
                sources=["transcript"]
            )
        )

    except Exception as e:
        print(f"\nError in extract_founder_data: {e}")
        if results:  # Only print results if they exist
            print("Results:", results)
        raise