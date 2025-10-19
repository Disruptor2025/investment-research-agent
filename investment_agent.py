# Final Project: Investment Research Agent
# Idrees Khan
# GitHub: https://github.com/Disruptor2025/investment-research-agent
# Date: 10/19/2025
"""
This is our final project for the Applied AI course. We built an autonomous
investment research agent that can analyze stocks using multiple AI agents
working together.

The system has three main workflow patterns:
1. Prompt Chaining - processes news articles step by step
2. Routing - sends tasks to the right specialist agent
3. Evaluator-Optimizer - generates analysis then improves it

We also built agent functions that plan, use tools, self-reflect, and learn.
"""

# Install required packages (run this first if you get import errors)
# !pip install yfinance requests pandas

import json
import os
from datetime import datetime, timedelta
import yfinance as yf
import requests
import pandas as pd

# =============================================================================
# PART 1: AGENT MEMORY SYSTEM
# =============================================================================
# This handles the "learns across runs" requirement. The agent stores insights
# from previous analyses and retrieves them when analyzing similar stocks.

class AgentMemory:
    """Stores and retrieves learnings from past analyses"""
    
    def __init__(self, memory_file="agent_memory.json"):
        self.memory_file = memory_file
        self.memories = self.load_memory()
    
    def load_memory(self):
        # load existing memories if the file exists
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        # otherwise start fresh
        return {
            "insights": [],
            "stock_history": {},
            "lessons_learned": []
        }
    
    def save_memory(self):
        # save to disk so it persists across runs
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f, indent=2)
    
    def add_insight(self, insight, stock_symbol=None):
        # store new learning
        entry = {
            "timestamp": datetime.now().isoformat(),
            "insight": insight,
            "stock": stock_symbol
        }
        self.memories["insights"].append(entry)
        self.save_memory()
        print(f"Stored insight: {insight[:50]}...")
    
    def get_past_learnings(self, stock_symbol):
        # get relevant memories for this stock
        learnings = []
        for insight in self.memories["insights"]:
            if insight.get("stock") == stock_symbol:
                learnings.append(insight["insight"])
        return learnings[-3:]  # just the last 3 to keep it relevant

# =============================================================================
# PART 2: TOOL INTEGRATION
# =============================================================================
# This covers the "uses tools dynamically" requirement. We integrate with
# Yahoo Finance API to get real stock data.

class FinancialTools:
    """Tools for fetching financial data"""
    
    @staticmethod
    def get_stock_info(symbol, period="1mo"):
        """Get stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info
            
            # extract the data we need
            data = {
                "symbol": symbol,
                "price": info.get('currentPrice', 'N/A'),
                "market_cap": info.get('marketCap', 'N/A'),
                "pe_ratio": info.get('trailingPE', 'N/A'),
                "52w_high": info.get('fiftyTwoWeekHigh', 'N/A'),
                "52w_low": info.get('fiftyTwoWeekLow', 'N/A'),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "recent_prices": hist['Close'].tolist()[-10:],
                "volumes": hist['Volume'].tolist()[-10:]
            }
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    @staticmethod
    def get_news(symbol):
        """
        Get news for a stock. 
        NOTE: We're simulating news here because real news APIs require paid keys.
        In production you'd use NewsAPI.org or similar.
        """
        # simulated news data for demo purposes
        news = [
            {
                "title": f"{symbol} beats earnings expectations for Q4",
                "content": f"Company {symbol} reported strong quarterly results with revenue growth of 15% year-over-year. Analysts are optimistic about future performance.",
                "source": "Financial News Daily",
                "published": (datetime.now() - timedelta(days=2)).isoformat(),
                "sentiment": "positive"
            },
            {
                "title": f"Analysts upgrade {symbol} rating to buy",
                "content": f"Several major investment banks have upgraded their rating on {symbol} citing improved market position and strong fundamentals.",
                "source": "Market Watch",
                "published": (datetime.now() - timedelta(days=4)).isoformat(),
                "sentiment": "positive"
            },
            {
                "title": f"{symbol} faces new regulatory challenges",
                "content": f"Government agencies are reviewing {symbol}'s business practices which may lead to new compliance requirements.",
                "source": "Business Insider",
                "published": (datetime.now() - timedelta(days=6)).isoformat(),
                "sentiment": "negative"
            }
        ]
        return news
    
    @staticmethod
    def get_market_overview():
        """Get overall market context using SPY as benchmark"""
        try:
            spy = yf.Ticker("SPY")
            data = spy.history(period="1mo")
            
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            change_pct = ((end_price - start_price) / start_price) * 100
            
            return {
                "market_trend": "bullish" if change_pct > 0 else "bearish",
                "sp500_change_pct": change_pct,
                "volatility": data['Close'].std()
            }
        except:
            return {"market_trend": "neutral", "sp500_change_pct": 0, "volatility": 0}

# =============================================================================
# WORKFLOW 1: PROMPT CHAINING
# =============================================================================
# This implements the required workflow pattern:
# Ingest News -> Preprocess -> Classify -> Extract -> Summarize

class NewsChainProcessor:
    """
    Processes news through a 5-step chain where each step feeds into the next.
    This is one of our required workflow patterns.
    """
    
    def __init__(self):
        self.results = {}
    
    def step1_ingest(self, news_list):
        """Step 1: Ingest raw news articles"""
        print("\nStep 1: Ingesting news...")
        self.results['raw_news'] = news_list
        print(f"Ingested {len(news_list)} articles")
        return news_list
    
    def step2_preprocess(self, articles):
        """Step 2: Clean and normalize the text"""
        print("Step 2: Preprocessing...")
        cleaned = []
        for article in articles:
            clean_article = {
                "title": article['title'].strip(),
                "content": article['content'].strip().lower(),
                "source": article['source'],
                "date": article['published'],
                "word_count": len(article['content'].split())
            }
            cleaned.append(clean_article)
        
        self.results['preprocessed'] = cleaned
        print(f"Preprocessed {len(cleaned)} articles")
        return cleaned
    
    def step3_classify(self, articles):
        """Step 3: Classify articles by type"""
        print("Step 3: Classifying...")
        
        for article in articles:
            content = article['content']
            
            # simple keyword classification
            if 'earnings' in content or 'revenue' in content or 'profit' in content:
                article['category'] = 'earnings'
            elif 'regulatory' in content or 'regulation' in content:
                article['category'] = 'regulatory'  
            elif 'analyst' in content or 'rating' in content or 'upgrade' in content:
                article['category'] = 'analyst'
            else:
                article['category'] = 'general'
        
        self.results['classified'] = articles
        print("Classification complete")
        return articles
    
    def step4_extract(self, articles):
        """Step 4: Extract key information from each article"""
        print("Step 4: Extracting key info...")
        
        for article in articles:
            # extract important data points
            article['key_info'] = {
                "main_topics": self._get_topics(article['content']),
                "sentiment": self._determine_sentiment(article),
                "importance": self._score_importance(article)
            }
        
        self.results['extracted'] = articles
        print("Extraction complete")
        return articles
    
    def step5_summarize(self, articles):
        """Step 5: Generate summary of all articles"""
        print("Step 5: Summarizing...")
        
        # count by category
        categories = {}
        for article in articles:
            cat = article['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        # calculate overall sentiment
        sentiments = [a['key_info']['sentiment'] for a in articles]
        pos_count = sentiments.count('positive')
        neg_count = sentiments.count('negative')
        
        if pos_count > neg_count:
            overall_sentiment = "POSITIVE"
        elif neg_count > pos_count:
            overall_sentiment = "NEGATIVE"
        else:
            overall_sentiment = "NEUTRAL"
        
        summary = f"""
News Summary:
Total articles: {len(articles)}
Earnings: {categories.get('earnings', 0)}, Regulatory: {categories.get('regulatory', 0)}, Analyst: {categories.get('analyst', 0)}, General: {categories.get('general', 0)}
Overall sentiment: {overall_sentiment} ({pos_count} positive, {neg_count} negative)
Top story: {articles[0]['title'] if articles else 'None'}
        """
        
        self.results['summary'] = summary
        return summary
    
    def _get_topics(self, content):
        """Helper to extract topics"""
        topics = []
        keywords = ['earnings', 'revenue', 'profit', 'growth', 'market', 'regulatory']
        for kw in keywords:
            if kw in content:
                topics.append(kw)
        return topics
    
    def _determine_sentiment(self, article):
        """Helper to determine sentiment"""
        # in real system would use NLP model
        return article.get('sentiment', 'neutral')
    
    def _score_importance(self, article):
        """Helper to score article importance"""
        score = 0.5
        if article['category'] == 'earnings':
            score += 0.3
        # check topic count from content directly
        topics = self._get_topics(article['content'])
        if len(topics) > 2:
            score += 0.2
        return min(score, 1.0)
    
    def run_chain(self, news_list):
        """Execute the full 5-step chain"""
        print("\nRunning news processing chain...")
        
        data = self.step1_ingest(news_list)
        data = self.step2_preprocess(data)
        data = self.step3_classify(data)
        data = self.step4_extract(data)
        summary = self.step5_summarize(data)
        
        print("Chain completed")
        return summary

# =============================================================================
# WORKFLOW 2: ROUTING
# =============================================================================
# This implements the routing workflow where we direct tasks to specialist agents

class SpecialistAgents:
    """Different specialist agents for different types of analysis"""
    
    @staticmethod
    def earnings_specialist(stock_data):
        """Analyzes earnings and financials"""
        print("  -> Earnings specialist analyzing...")
        
        pe = stock_data.get('pe_ratio', 'N/A')
        mcap = stock_data.get('market_cap', 'N/A')
        
        analysis = f"""
Earnings Specialist Analysis:
- P/E Ratio: {pe}
- Market Cap: {mcap}
- Assessment: """
        
        if pe != 'N/A' and pe < 15:
            analysis += "Potentially undervalued based on P/E"
        elif pe != 'N/A' and pe > 30:
            analysis += "May be overvalued, monitor closely"
        else:
            analysis += "Fairly valued in current market"
        
        return analysis
    
    @staticmethod
    def news_specialist(news_data):
        """Analyzes news and sentiment"""
        print("  -> News specialist analyzing...")
        
        pos = sum(1 for n in news_data if n.get('sentiment') == 'positive')
        neg = sum(1 for n in news_data if n.get('sentiment') == 'negative')
        total = len(news_data)
        
        analysis = f"""
News Specialist Analysis:
- Total articles: {total}
- Positive: {pos}
- Negative: {neg}
- Sentiment: """
        
        if pos > neg:
            analysis += "Bullish news sentiment"
        elif neg > pos:
            analysis += "Bearish news sentiment"
        else:
            analysis += "Neutral news sentiment"
        
        return analysis
    
    @staticmethod
    def market_specialist(market_data, stock_data):
        """Analyzes market context"""
        print("  -> Market specialist analyzing...")
        
        trend = market_data.get('market_trend', 'neutral')
        sector = stock_data.get('sector', 'Unknown')
        
        analysis = f"""
Market Specialist Analysis:
- Overall market: {trend}
- Stock sector: {sector}
- Context: """
        
        if trend == 'bullish':
            analysis += "Positive market environment supports growth"
        elif trend == 'bearish':
            analysis += "Challenging market conditions, caution advised"
        else:
            analysis += "Mixed market signals"
        
        return analysis


class TaskRouter:
    """
    Routes tasks to the appropriate specialist agent.
    This is our second required workflow pattern.
    """
    
    def __init__(self):
        self.specialists = SpecialistAgents()
    
    def route(self, query, data):
        """Determine which specialists to use based on the query"""
        print("\nRouting to specialists...")
        
        results = []
        query_lower = query.lower()
        
        # route based on query keywords
        if 'earnings' in query_lower or 'financial' in query_lower:
            results.append(self.specialists.earnings_specialist(data['stock']))
        
        if 'news' in query_lower or 'sentiment' in query_lower:
            results.append(self.specialists.news_specialist(data['news']))
        
        if 'market' in query_lower or 'sector' in query_lower:
            results.append(self.specialists.market_specialist(
                data['market'], data['stock']))
        
        # if no specific routing detected, use all specialists for complete analysis
        if not results:
            print("Using all specialists...")
            results.append(self.specialists.earnings_specialist(data['stock']))
            results.append(self.specialists.news_specialist(data['news']))
            results.append(self.specialists.market_specialist(
                data['market'], data['stock']))
        
        print("Routing complete")
        return results

# =============================================================================
# WORKFLOW 3: EVALUATOR-OPTIMIZER
# =============================================================================
# This implements the generate -> evaluate -> refine workflow

class AnalysisEvaluator:
    """
    Evaluates analysis quality and refines it iteratively.
    This is our third required workflow pattern and also covers "self-reflects".
    """
    
    def __init__(self, max_iterations=3):
        self.max_iterations = max_iterations
        self.history = []
    
    def generate(self, stock_data, news_data, iteration=1):
        """Generate analysis"""
        if iteration == 1:
            print("\nGenerating analysis...")
        
        symbol = stock_data.get('symbol', 'UNKNOWN')
        price = stock_data.get('price', 'N/A')
        pe = stock_data.get('pe_ratio', 'N/A')
        
        # count news sentiment
        pos_news = sum(1 for n in news_data if n.get('sentiment') == 'positive')
        total_news = len(news_data)
        
        # generate the analysis
        analysis = f"""
Stock: {symbol}
Price: ${price}, P/E: {pe}, Sector: {stock_data.get('sector', 'N/A')}
News: {pos_news}/{total_news} positive articles
Recommendation: """
        
        # make recommendation based on data
        if pos_news > total_news/2 and pe != 'N/A' and pe < 25:
            analysis += "BUY - Positive sentiment and reasonable valuation"
        elif pos_news < total_news/2 or (pe != 'N/A' and pe > 30):
            analysis += "HOLD - Mixed signals, wait for better entry"
        else:
            analysis += "NEUTRAL - Monitor for changes"
        
        return analysis
    
    def evaluate(self, analysis):
        """Evaluate the quality of the analysis"""
        
        # check for key components
        has_recommendation = 'Recommendation:' in analysis
        has_price = 'Price:' in analysis
        has_pe = 'P/E' in analysis
        has_news = 'News' in analysis
        has_reasoning = any(word in analysis for word in ['because', 'due to', 'based on'])
        
        # calculate quality score
        score = sum([
            has_recommendation,
            has_price,
            has_pe,
            has_news,
            has_reasoning
        ]) / 5.0
        
        # generate feedback
        feedback = []
        if not has_recommendation:
            feedback.append("Add clear recommendation")
        if not has_reasoning:
            feedback.append("Include more reasoning")
        if score < 0.6:
            feedback.append("Add more detail and context")
        
        passed = score >= 0.6
        
        return {
            "score": score,
            "passed": passed,
            "feedback": feedback
        }
    
    def refine(self, analysis, feedback, stock_data, news_data):
        """Refine analysis based on feedback"""
        
        # add missing components based on feedback
        refined = analysis
        
        if "more reasoning" in str(feedback):
            refined += "\n\nReasoning: This recommendation is based on technical analysis and news sentiment."
        
        if "more detail" in str(feedback):
            refined += f"\n\nAdditional context: Stock is in {stock_data.get('sector', 'unknown')} sector."
        
        return refined
    
    def run_loop(self, stock_data, news_data):
        """Run the full evaluate-optimize loop"""
        
        current = self.generate(stock_data, news_data, 1)
        
        for i in range(1, self.max_iterations + 1):
            eval_result = self.evaluate(current)
            
            self.history.append({
                "iteration": i,
                "analysis": current,
                "evaluation": eval_result
            })
            
            if eval_result['passed']:
                break
            
            if i < self.max_iterations:
                current = self.refine(current, eval_result['feedback'], 
                                     stock_data, news_data)
                current = self.generate(stock_data, news_data, i+1)
        
        return current

# =============================================================================
# MAIN INVESTMENT RESEARCH AGENT
# =============================================================================
# This is the main agent that ties everything together

class InvestmentAgent:
    """
    Main autonomous investment research agent.
    Demonstrates all required agent functions:
    - Plans research steps
    - Uses tools dynamically  
    - Self-reflects on quality
    - Learns across runs
    """
    
    def __init__(self):
        self.memory = AgentMemory()
        self.tools = FinancialTools()
        self.news_chain = NewsChainProcessor()
        self.router = TaskRouter()
        self.evaluator = AnalysisEvaluator()
    
    def plan_research(self, stock_symbol):
        """
        Create a research plan for analyzing a stock.
        This covers the "plans its research steps" requirement.
        """
        print(f"\nPlanning research for {stock_symbol}...")
        
        # check if we have past learnings about this stock
        past_learnings = self.memory.get_past_learnings(stock_symbol)
        if past_learnings:
            print(f"Found {len(past_learnings)} past learnings about {stock_symbol}")
        
        # create the research plan
        plan = [
            f"1. Fetch stock data for {stock_symbol} from Yahoo Finance",
            f"2. Gather recent news articles",
            f"3. Get overall market context",
            f"4. Run news through prompt chaining pipeline",
            f"5. Route analysis to specialist agents",
            f"6. Generate and optimize final analysis",
            f"7. Self-reflect on quality",
            f"8. Store learnings for future use"
        ]
        
        print("Research plan created")
        return plan
    
    def research(self, stock_symbol):
        """
        Execute research on a stock using all workflows and agent capabilities.
        This is the main method that demonstrates everything.
        """
        print(f"\nResearching {stock_symbol}...")
        
        # STEP 1: Planning
        plan = self.plan_research(stock_symbol)
        
        # STEP 2: Tool usage - gather data
        print("Gathering data...")
        stock_data = self.tools.get_stock_info(stock_symbol)
        news_data = self.tools.get_news(stock_symbol)
        market_data = self.tools.get_market_overview()
        
        print(f"Got stock data, {len(news_data)} news articles, and market data")
        
        # STEP 3: Workflow 1 - Prompt Chaining
        news_summary = self.news_chain.run_chain(news_data)
        
        # STEP 4: Workflow 2 - Routing
        query = f"comprehensive analysis of {stock_symbol}"
        routing_results = self.router.route(query, {
            'stock': stock_data,
            'news': news_data,
            'market': market_data
        })
        
        # STEP 5: Workflow 3 - Evaluator-Optimizer
        final_analysis = self.evaluator.run_loop(stock_data, news_data)
        
        # STEP 6: Self-reflection
        print("\nSelf-reflection...")
        reflection = self.reflect(stock_data, news_data, final_analysis)
        
        # STEP 7: Learning - store insights for next time
        insight = f"Analyzed {stock_symbol} - sector: {stock_data.get('sector', 'unknown')}, sentiment: {'positive' if len([n for n in news_data if n.get('sentiment')=='positive']) > len(news_data)/2 else 'mixed'}"
        self.memory.add_insight(insight, stock_symbol)
        
        # compile everything into final report
        report = {
            "symbol": stock_symbol,
            "plan": plan,
            "data": {
                "stock": stock_data,
                "news_count": len(news_data),
                "market": market_data
            },
            "news_summary": news_summary,
            "specialist_analyses": routing_results,
            "final_analysis": final_analysis,
            "reflection": reflection,
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def reflect(self, stock_data, news_data, analysis):
        """
        Self-reflection on research quality.
        This covers the "self-reflects" requirement.
        """
        reflection = f"""Quality check: Data is {'complete' if stock_data.get('price') != 'N/A' else 'incomplete'}, analyzed {len(news_data)} articles, final score {self.evaluator.history[-1]['evaluation']['score']:.2f}"""
        
        return reflection

# =============================================================================
# EXECUTION
# =============================================================================
# This is where we actually run everything

def main():
    """Main function that runs the agent and demonstrates all requirements"""
    
    print("\nStarting Investment Research Agent...")
    print("Analyzing stocks: AAPL, MSFT\n")
    
    # create the agent
    agent = InvestmentAgent()
    
    # test with a couple stocks to show learning across runs
    stocks = ["AAPL", "MSFT"]
    reports = []
    
    for stock in stocks:
        print(f"\n--- Analyzing {stock} ---")
        
        report = agent.research(stock)
        reports.append(report)
        
        # print summary
        print("\nResults:")
        print(report['news_summary'])
        print("\nSpecialist analyses:")
        for i, analysis in enumerate(report['specialist_analyses'], 1):
            print(f"\n{i}.{analysis}")
        print("\nFinal recommendation:")
        print(report['final_analysis'])
        print("\n" + "-"*50)
    
    # save reports
    print("\n\nSaving research reports...")
    with open('research_reports.json', 'w') as f:
        json.dump(reports, f, indent=2, default=str)
    print("Saved to research_reports.json")
    
    print("\n\nDone! Check research_reports.json for full results.")
    
    return reports

if __name__ == "__main__":
    results = main()
