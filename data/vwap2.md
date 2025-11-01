1 PREFACE
An untold truth is, that most published strategies don’t make money. Most
of them work only on circumstances which occur very seldom. The VWAP
Strategy or generally the Equilibrium Price Strategy works every day.

I wrote a VWAP indicator for Metatrader 5 (Mql5.com). I choosed  a slim
version and display VWAP only for the actual day. Other implementations
which run over the whole displayed period crashed Metatrader.

The below equity curve stems from the first step of qualification of TopStep
Traders (funded trading). I reached $ 9000 in 22 days trading the Mini Dow
Future only. It was achieved mainly by watching equilibrium prices
(VWAP, Pivot Points, pre day lines, Volume Priofile) and watching trends in
the minute chart by drawing trend lines. It shows that it is possible to get a
positive expectation value even in an index with a high percentage of high
frequency trading.

Fig. 1: Equity curve of Dow Jones Mini Futures trading by applying the
strategy of equilibrium prices (first step of TopStep Traders to get a funded
account).

WVAP is simply a price weighted by volume. It marks a fair price, i.e. a
price where you are not supposed to trade.

WVAP works best with liquid assets as stock indices or blue chips, but also
for currencies, but only if there is volume data present and the volume must
be large enough. The strength is that VWAP is independent of time frame
and it helps to reveal institutional activities. A drawback is that it needs
precise volume data and there are different values for different  starting
times. In addition VWAP can be slightly different for related assets, eg.
each VWAP of Dow Jones Cash, Mini Future and Micro Future differ. But
every VWAP has its own value.

2 DISCLAIMER &
COPYRIGHTS

Trading indices and currencies involves substantial risk. Losses are always
part of the game. The advices in this book will not guarantee profits from
trading. There is no trade recommendation or investment advice. All
content in this book is for teaching and informational purposes
only.

Before deciding to invest in the financial market, you should
carefully consider your investment objectives. Your level of
experience and your risk threshold. Don’t invest money you cannot
afford to lose. It is a fact that more than 80% of the European retail
trader lose.

Your success using the strategy of this book depends on many
factors. The author  has no way of knowing how well you will do.
Therefore he does not guarantee or imply you will get any earnings
at all.

The short term trading presented here needs substantial routine to
be successful. It is viable that you understand when the situation in
the market does not allow trading.

All trademarks are trademarks of their respective owners. We use
names in an editorial fashion only.

3 TIME
I use CET (Central Europe Time) and EST (Eastern Time US). The time
difference is 6 hours in normal situations. During summertime conversion
this difference reduces to 5 hours. There might be a time jam in the depicted
charts as brokers normally don’t change their server time during the three
weeks when US has already summer time and Europe does not have it yet.
The same is valid for autumn.

VWAP has a different appearance for different starting times. Check for
your timezone if price bounces often on VWAP or falls often back to
VWAP. If so, your VWAP is correct.

4 INTRODUCTION
VWAP stands for V olume W eighted A verage P rice. VWAP shows an
equilibrium price, i.e. a fair price, and this price is more dynamic than for
example the point of control of a market- or volume profile. VWAP has the
big advantage of not being dependent of timeframe. Whatever timeframe
you chose, VWAP is the same. There are only few indicators being
independent of time frame, as the Pivot Points for example. As VWAP
needs precise volume data the displayed curve might not be very precise. If
you compare the VWAP representation of Dow Jones Cash, Dow Jones
Mini- and Micro-Futures there are some differences in representation. In
such a case you have to watch the VWAP with the market of the largest
volume, i.e. Dow Jones Cash market.

The formula of VWAP is

As you see there are very big numbers involved during the day. In the
course of the day it may get less precise because of rounding errors. Many
implementations may crash Metatrader because of the many calculations
needed. Only one single error in volume data may shift the curve. So you
must not look at VWAP at a line, rather think of it as a zone. In the figure
below there is a typical error: In the left side I traded short when price
dipped below VWAP in a TradingView representation. When I looked at the
picture after the close of the stock exchange there was another picture. It is
probably due to bad volume data. For a precise volume data stream you
have to pay extra in TradingView. In any case VWAP is no signal line, it is
a region of a fair price.

Fig. 4.1: Depending on the quality of the volume data VWAP might not be
very precise. On the right the representation of the moment I traded. On the
right side the representation after the close of the stock exchange. The
signal at the left side was wrong. In addition it is important when VWAP
starts. Depending on the starting time you get different curves.

VWAP divides a trading day in a long bias and short bias zone. If you orient
to VWAP during trading you may foresee some  strong rebounds or
pullbacks. If price is above VWAP and you short, the danger of a sudden
rebound is much larger as if the price was below VWAP and vice versa for
the long case. By its nature as day divider you should be flat if price is near
VWAP and you may not trade a cross of price with VWAP.

VWAP also may reveal institutional activity. And if you manage to spot
institutional activity you are the king of the arena.

You can get the Swiss VWAPsimple for free from my website on
mql5.com. It runs on Metatrader 5. Although I use only tick volume in the
calculations this indicator works very well, at least for indices as German
Dax or Dow Jones.

This book doesn’t explain every indicator, you should be familiar with
them. If not please read one of my other books “The Opening Range
Oriented Strategy”.

This book is a working book and will have several releases, please check if
you have the up to date version.

Actual version: 1.1.

5 SOURCES TO LEARN
FROM
There are many Youtube films or Facebook Sites which suggest you to buy
an expensive crash course to learn trading. They mostly show very nice
single trades and the life style of the instructors (mainly luxury cars),
absolutely not related to trading. Don’t believe these advertisements and
don’t spend money on them. First you have to find your own way of trading
and second you cannot trust single trades. The only proof that somebody
can consistently trade is an equity curve, and even an equity curve can be
forged.

When starting trading you should concentrate mainly on the US-Trading
veterans. These true gurus don’t show you their live style. And don’t think
they are outdated. They are not at all, the principles they outline are still
valid. First Linda Raschke. If you listen to her in Youtube you get more
trading truth than from anywhere. She also tells you which literature you
should read. A problem might be, that you don’t understand the true gurus
in the beginning due to too few knowledge about trading. So it is worth to
spend some practice in trading to understand them. Another very important
author is James Dalton. His book Mind over Markets is a corner stone in
understanding equilibrium prices.

In Youtube there are many films about trading that want to sell some course
or signals to you. Sometimes there is a really good traders behind these
films, but they would never tell you the total truth behind their amazing
trades they present as they want to make money with their knowledge. With
some experience you will be able to read between the lines and see what is
behind the presented trades. I learned a lot by analysing such
advertisements.

Don’t listen to dayli financial entertainment (fintainment) which make a
forecast of markets every day. They are sponsored by the trading industry
and have conflicts of interest. This sort of entertainment will distract und
unsettle your thoughts.

Some very important literature to read

●          James Dalton, Mind over Markets (how markets work, market

profile)

●          Linda Raschke, Street Smarts (many strategies)
●          Crabel Toby, Day Trading With Short Term Price Patterns (Opening

Range Breakout strategy)

●         Velez Oliver, Tools and Tactics for the Master DayTrader: Battle-

Tested Techniques for Day, Swing, and Position Traders (Day trading
of shares)

●          Charles le Beau & David Lucas, Day Trading Systems & Methods

Some interesting authors in Youtube:

●          Linda Raschke (Strategies, how to trade)
●          Oliver Velez (Trading stocks, Trading psychology)
●          John Piper (Elliott Waves)
●          Mark Douglas (Trading Psychology)
●          Alexander Elder (trading in general)
●          Chat With Traders (Aaron Fifield, intverviews with traders)

Don't forget to read my other book: The Opening Range  Oriented Strategy.

My Youtube channel:

Swiss Trader
https://www.youtube.com/channel/UCljXBKcASRPXEDNootJl0rg

6 PRINCIPLES OF VWAP
VWAP is one of many equilibrium prices. You should only trade if the price
has some distance from the equilibrium. If the price is at equilibrium there
is a danger of being in a narrow range.  Also signals created by any
equilibrium price, especially VWAP, e.g. crossing with EMA20, have to be
interpreted very carefully, as VWAP is no exact signal line.

The meaning of VWAP has to be seen an a global context. If price dips
below VWAP in a global uptrend it is something else than if in a global
downtrend.

VWAP daily as we use it in our strategy starts always from midnight (00:00
CET). The current sum of price times volume is divided by the sum of
volume. VWAP is independent of timeframe, due to large numbers and
small errors that add, VWAP may differ slightly in another time frame. If
you trade US-Indices on a US-Platform VWAP will be started 6 hours later.
You should watch especially the VWAP which started at midnight of
Eastern Time.

Trading the VWAP: The VWAP-price can be looked at as fair. A fair price
is not interesting for a trader. In the following figure you see, that it is best
entering a new trend after forming an ABC, as far away from the VWAP as
possible.

Fig. 6.1: VWAP divides the day in a long and a short region as do for
example the point of control of the volume profile, the pivot points or long
EMAs and SMAs. Trading near the day divider is always risky as you trade
at a fair price and a fair price is either cheap nor expensive. You should
never go short at a near distance above VWAP and never go long at a near
distance below VWAP. The chance of a bear or bull trap is big. If you go
short over the VPAP you have to look if there is an asymmetric opportunity
for take profit still over the VWAP, as VWAP is a mayor resistance. If you
trade short a fallback of price to VWAP place your take profit near above
VWAP.

As depicted in the figure above look to trade long only above vwap and
short only below VWAP. Before the US-Opening watch the VWAP which
started at European midnight. After the open of the US-Markets watch the
VWAP from US-Midnight (0:00 am ET). You may only go short in the long
bias zone of VWAP if the distance to VWAP is big and you see at least a

double top in the five minutes chart and an evolving new trend. An
asymmetric opportunity must exist within the distance over VWAP.

Fig. 6.2: VWAP is a sort of fair price (violet stars, Swiss VWAPsimple).
Therefore you should not trade if price is near VWAP. In this figure VWAP
is used as a level to take profit. Ths long trade is started as far away as
possible from VAP. If you spot a new trend you should enter as early as
possible and exit near VWAP.

Fig. 6.3: Probable institutional selling. Price bounces of VWAP
(violet). It is wrong to go short at such a situation because the
distances are far too small for a retail trader. Just wait until the

elephant has sold its shares. What follows is a short squeeze, i.e.
the price runs up.

Fig. 6.4: Typical behaviour of VWAP (thick violet line). In 1 there are many
rebounds to the VWAP from 19:00 to 20:50 CEST. At 20:50 there is a short
squeeze when price breaks over VWAP and the big sellers vanished. At
21:50, ten minutes before close, price falls back to VWAP (Friday July 26,
2019).

The Swiss VWAPsimple (download for free form mql5.com) has proved to
be mostly correct. There are many implementations which show wrong
numbers. This indicator depends strongly on the precision of volume.
Although there is tick volume only in Metatrader it seems to be quite
precise. As any little error in volume data may shift the curve, you should
watch the region around the curve, not exactly the line.

Fig. 6.4: Price orienting to VWAP on a bearish day. Price crossed VWAP at
08:15 CEST and never came back over VWAP, but touched VWAP several
times. 29 May 2019.

In the figure below you see the Tiki-Vwap with standard deviation bands.

Fig. 6.5: Price of VWAP in Mini-Dow Jones Future lies about 50 points
higher than the VWAP of Dow Jones Cash started at 0:00 CEST. The
reason might be the other volume of this asset or a slightly different starting
time. In such a case watch both VWAPs. The VWAP with the most matching
points is the best one. Be sure that the beginning time of VWAP calculations
are the same if you watch different VWAPs. Representation by Tradovate
with Tiki-Vwap.

Fig. 6.6: Dow Jones rebounds to VWAP.  As VWAP can be looked a a main
resistance price might often bounce back to vwap. This effect is enforced as
a whole hundred number was crossed (26100).

Fig. 6.7: Rebound of price at VWAP. If you trade short over VWAP place
your take profit still above VWAP. Don’t try to trade a cross of price with
VWAP (August 21 2019).

Fig. 6.8: Typical rebound from VWAP in Dow Jones index in the last
trading hour. If you are short place the take profit near above the VWAP.

7 THE GLOBAL CONTEXT
The global context is very important, before trading you must know if your
market is in a global up-, a downtrend or a sideways range.

Fig. 7.1: Before trading in the morning watch the global trend. By
displaying the volume profile and comparing the points of control you see
exactly in what situation the market is.

Fig. 7.8: 14 Jun 2019, price can pop to VWAP in a sideways range.

In a global uptrend a dip below VWAP may reveal a long chance, but in a
global downtrend it is a probable short opportunity.

The following Pine Script for TradingView shows crosses from VWAP with
price (represented by a simple moving average of 3). Don’t trade the
crosses but the rebounds or pullbacks to VWAP are more secure to trade. In
addition the script is very slow and therefore not suited for a reliable signal.

//@version=2
//Not suited for trading, signals show late
//Use in 1 minutes chart
strategy(title="VWAPStrategy", overlay=true)

//Values for Simple Moving Average fast
valfast=3
//Value for VWAP
valslow= vwap(hlc3)

long = sma(close, valfast)
short = valslow

plot(short, color = red)
plot(long, color = green)

longCond=crossover(long,short)
shortCond=crossover(short,long)
strategy.entry("LONG", strategy.long, when=longCond)
strategy.entry("SHORT", strategy.short, when=shortCond)
strategy.exit("EXLONG", from_entry="LONG", when=shortCond)
strategy.exit("EXSHORT", from_entry="SHORT", when=longCond)

Fig. 7.9: Pine script to show crosses of price with VWAP to get a global
overview. Don’t trade crosses with VWAP directly as rebounds and
pullbacks occur very often near VWAP.

8 THE STRATEGY
You don’t get any buy or sell signals by VWAP. The best thing to watch
changing trends is to draw trend lines constantly and watch if they are
broken. It is the fastest way to get entry signals. Try to spot impulsive
Elliott waves by spotting double or triple bottoms or tops.

You can also use any indicator, but most of them are very delayed.  You
also can use for example the direction change of Parabolic SAR (very clear
signals, but dependent on the parameters and time frame) or the
oversold/overbought signals from RSI combined with the same signals of
Stochastics. Signals can also be used from any crossing lines as moving
averages or Tenkan and Kijun Sen of the Ichimoku indicator or the +Di
with -Di from the ADX indicator. I personally don’t like the signals of
crossing lines as they can be ambiguous if the two lines cross in a flat angle
and cause trouble when programming the strategy.

If price is over the VWAP trade only long, ignore short signals and vice
versa. You only may go short in the long bias zone when price is really far
from VWAP, best at an extreme point, and vice versa and there is a new
trend evolving in the counter direction. If you have a VWAP indicator with
standard deviation bands go short only if price is at least over the first
standard deviation. As you don’t trade during a cross of price with VWAP
you may lose some opportunities, but it is a small price for the security you
win.

VWAP is used as a filter.

●          Don’t trade if price is near VWAP, ignore any other signals.
●         D on’t trade crosses from WWAP with price, look for pullbacks or

rebounds to VWAP.

●          Beware of bear- and bull-traps near VWAP.
●          If price is near above VWAP don’t trade short, trade long.
●          If price is near below VWAP don’t trade long, trade short.

●         I f price is far above VWAP you may trade a new short trend. Bear

in mind that you trade against the global trend.

●         I f price is far below VWAP you may trade a new long trend. Bear

in mind that you trade against the global trend.

●          Ask yourself if there is enough space for an asymmetric

opportunity (Risk-Reward 2:1)?

“Far above VWAP” means at least one standard deviation or better two
standard deviations distance from price.

Beware of heavy rebounds if you trade Dow Jones. These rebounds mainly
occur near VWAP or near 25, 50, 75, 100 and 1000 levels.

If there is a global long trend and price dips below VWAP there might be a
long rally soon.

Fig. 8.1: Dow Jones dipping below VWAP in a global uptrend. Dow forms a
double bottom in form of a Elliott 1 and 2 wave. Wave 3 follows. Wave 4 is
different from wave 2 and the last wave is in form of an ending diagonal.
Look for Elliott impulsive waves when trading.

To spot a new trend you may use the Swing Chart indicator in the 1 to 10
minutes chart. Swing Chart draws candles bull, bear, inside and outside
bars. It also draws lines between the candles. It is very useful to have the
global overview if you use it in a 10 minutes chart. The lines in between the
candles show you if the trend is still intact.

In Metatrader 5 you can download Swing Chart from Andre S. Enger for
free.

Fig. 8.2: Trading with the help of Swing Chart indicator in a 10 minutes
chart (30 May 2019, 09:52). The 10 min chart with the drawn trend swings
help you to stay in a trade and prevents you from a too early exit.

Fig. 8.3: With the help of the Fibo Retracement you can decide when to exit
a trend. A trend is still valid after a 100 % pullback. Here there was a
pullback to the 61.8 % level. So I stayed in the trade (30 May 2019, 10:15
CEST).

Fig. 8.4: Trading in a half hour chart with volume profile and Tiki-Vwap
with bands. The half hour chart is suited for the volume profile. It helps you
to reduce the numbers of trades, too.

9 KNOWING WHEN NOT
TO TRADE
A quote from James Dalton’s Mind over Markets:

“Many knowledgeable professionals estimate that markets trend only 20 to
30 percent of the time. Failure to recognize this fact is one of the main
reasons why a large number of traders don't make money”.

It is very important not to trade if there is no trend or no movements.
Trading in a range only works if the range is enough wide. Many times if
price is in a range it does not know where to go but there might be sudden
spikes where the direction is unpredictable. For Dow Jones you need about
60 and for the German Dax about 40 points of movement before you can
trade profitably. May be it is best to wait for such a minimal movement
before you start. Or you start only with trading after you spot some Elliott-
Wave like structure.

Don’t either trade if the price tries to change direction. Wait for a distinct
new trend.

Evaluate your trading journal for the best hours to trade. Be aware that there
are the Asian, the European and the US-Session with their own
characteristics of price movements. There are also timespans during the day
where you should not be invested due to jittery prices. It is often from 13:00
to 15:00 CEST (7 am to 9 am EST) where you may have chaotic price
movements by a variety of news. Be very careful if you trade at session
borders (Asian-, European- and US-Session).  Abrupt changes may occur.

Ranging markets

It is key to see when a market ranges sideways and is not tradable.
Although the indicators VWAP, ADX, Volume Profile and Ichimoku may
show that you are in a range, it is not always obvious from the beginning

on. Another problem is, that big movements often start from sideways
phases. It is sometimes hard to see the start of a new trend out of a range.
In the figure below the mini Dow (YM) was very well tradable until the
opening of the us futures. Then the chaos started.

Fig. 9.1: Chaotic sideways market after the open of Dow Jones. Market
ranges from 26500 to 26550. 29 April 2019. Tradovate, demo of
TopStepTrader.

Fig: 9.2: The whole movements of Dow get clearer only from 18:00 CET
(10:00 ET) on at 29 April 2019. It is best to trade Dow Jones only if it goes
straight up after the opening. The ADX indicator was below 20 from the
opening on.

Volatility means a degree of movement. Volatility can be measured by the
ADX-Indicator. With low Volatility there is no trend and movements are not
predictable. The ADX depends on the Timeframe, usually if no timeframe
is indicated it is implicitly the five minute timeframe is assumed. It is better
to watch ADX than volume alone. The volume might be big and there is no
movement if price is near the fair price.

Fig. 9.3: Low Volatility Situation on Friday April 5, 1019 at 13:30 CET,
right side of the chart. The whole week had rising figures, but on Friday the
market is saturated. At the end of the week there is no steam any more and
the Nonfarm Payrolls News start in one hour. The price stays beneath
VWAP (violet) and the Ichimoku cloud, the ADX (below) is lower than 25.
The opening Range measures only 17 Points. One other reason the market
does not move is that Dow Jones stays over a hundred level (26400). It is
key to understand these circumstances and not to trade.

Fig. 9.4: Mind the piano keyboard chart. If you see five minute candle like
these you should stay away from this market.

The characteristics of sideways markets are

●          Price often near VWAP, Point of Control or other equilibrium

prices

●          Price in value area of the volume profile
●          Price in or near Ichimoku cloud
●          Price stays the whole day in the opening range (the span of the first

hourly candle)
●          ADX below 25
●          Symmetric Triangles
●          Inside- and outside-candles near each other
●          Many crossings of ema10 and ema20
●          Price near 100 or 1000 in a dominant market (Dow Jones)
●          The five minute chart looks like a piano keyboard
●         The value area is relatively small (should be about 100 for Dow

Jones or 40 points for German Dax)

Most traders lose as they cannot see when a market ranges. A range phase
after a trend may eradicate all prior wins. One remedy is to avoid ranging
and choppy markets. So don’t trade indices directly after the US opening.
Only trade them if Dow and S&P go straight up or down.

If you happen to get into a ranging market it is best to look at a longer
timeframe e.g. 15 or 30 minutes instead of 5 minutes. You will see the
global direction better.

A pattern which is often seen in ranging markets of Dow Jones is the ping
pong from the VWAP of the cash market to the VWAP of the future market.
Both the VWAPs have their special meaning.

Choppy markets

Besides ranging markets there are spiking ones. They mostly occur at news
times, but they can occur at any time. May be even out of a Trump tweet.

Fig. 9.5: Extreme jitter at 14:50 CEST. There is no special explanation for
it. But usually the time from 13:00 to 15:30 CEST is not good to trade, as
markets may be very choppy as they only have a low volume before the
opening of the cash market.

Fig 9.6: Dow Jones chart showing a rebound form a thousand (25000) of
110 points. May 29, 2019. 10 s chart.

Usually the markets run better at times when cash markets are opened. The
pre open is always more vulnerable to spikes, which cannot be foreseen.

Asymmetric opportunity

When trading you always should have at least a risk reward ratio of 2:1. Say
you have a stop loss of 10 points so your overall trading distance measures
30 points. So if you want to go short above VWAP and VWAP is your main
resistance  you need at least a distance of 30 points from VWAP. You
cannot force the principle of an asymmetric opportunity. Just look to secure
your position as early as possible and trail the stop loss as far as possible.

Fig. 4.11: VWAP builds a solid support. If you want to go short in this
situation you can be sure that price will not go below VWAP easily. There is
not enough space to place a short trade, as if you have a stop loss of 10
points the whole trade takes a distance of 30 points, which is not there. No
trade because of a lacking asymmetric opportunity.

10 BACKTESTING

When trading liquid assets you compete with very fast algorithms.
Computers are much faster than the human brain, but never as clever and
they follow a rigid scheme. Many of the most sophisticated algorithms lost
a lot of money. It is mandatory to take notes in a journal while trading. You
must not trade with real money until your strategy proved to be good. If you
specialise on one market and you know exactly when to trade and when not
you may be by far better than all the algorithms together. Don’t forget that
even algorithms fight against each other and form patterns which can be
recognised by the human brain.

With backtesting I understand to resimulate past trading days and test old
and new rules and not to develop a multivariate super robot that trades
automatically.

In the figure below I retest the past trading days and look how VWAP and
price go together with the Swiss Backtester EA , a simulation tool for
Metatrader 5 you get for free by me (see resources). You can also
resimulate a trading day with the replay function of TradingView.com.

Fig. 10.1 Backtesting a choppy preopen behaviour of Dow Jones of 6 June
2019 with the Swiss ImportantSignals EA.

11 PRACTICE
Trendlines are very important while trading, I cannot stress it enough. They
show signals faster than any indicator. You should always draw trendlines
in the hourly, the five and the one minutes chart. You have to draw them by
hand as automatically drawn lines may irritate. Don’t start a trade without
drawing a trendline. Trade along a trend line or a break of it.

Fig. 11.1: Drawing trendlines in the one minutes chart helps to understand
the mechanism of the Market. Many markets orient to trendlines (June 22,
2019).

The hourly chart is very important, too. In the figure below price runs over
the VWAP. The Volume of the peak is eliminated by three hammer candles
as short traders have to cover. Ten minutes before close the price falls to
VWAP and lower.

Fig. 11.2: Spike at 17:00 on June 22, 2019. The volume has to be reduced
again. At least at 21:50 the price falls below VWAP.

Bear in mind that VWAP may be a main support. Don’t short directly over
VWAP if there is not enough room for a trade. If your risk R is 10 points
(stop loss 10 points)  you need 30 points for your trade above VWAP. But

don’t go short without a double top and a starting new downtrend in the five
minutes chart.

Fig. 11.3: No trade, even if you have space. If your R is 10 points your trade
takes at least 30 points. As the distance to the VWAP is more than 30 points
you may risk a trade. But you don’t have much space left. But space is not
enough. You must have at least a double top, and a new downtrend starting.

Fig. 11.4: Double top of former figure did not evolve. You have to wait for a
potential next double top.

Fig. 11.5 Only after the open of the European markets there might be a
small chance for a short. But you should wait 15 Minutes before trading
(20.6.19)

Fig. 11.6: When day progresses draw the main support/resistance lines.
(20.6.18, 14:43 CET)

Fig. 11.7: Bearish day in Dow Jones, 12 June 2019, 12:50 CEST: Price
visits VWAP again and again. In this case you should short beginning at
VWAP (violet line above the price), but only if you see a new trend.

It is very difficult to trade without movements. One measure to check if
there is movement is to measure the height of the value area.

Fig. 11.8: Measuring the height of the value area gives a hint, if an asset
can be traded. German Dax only measures 30 points. (12 June, 2019, 16:33
CEST). 30 points is hardly enough to trade with wins. Dow Jones measures
at the same time 66 points. For good trades you need about 40 points with
Dax and about 100 points with Dow Jones.

Fig. 11.9: Price breaks over VWAP and the opening range (light blue lines)
at 9:15 as PMI data of France are very good. Following the VWAP Strategy
rules it would not be allowed to trade the break trough of the VWAP in long
direction with an unsecured position. After the head and shoulder pattern
the price falls back to the opening range. In trading short you have a short
pause when price is a VWAP as there is always the possibility of a rebound
at the VWAP (August 22, 2019).

The evaluation of your trading journal is important. Build a Playbook for
your successful trades.

Good Days:

Fig.  11.10:  May 7, 2019, very good day. Crossdown of price below VWAP.
Clear downtrend.

Fig.  11.11: May 22, 2019: Very good day, price bounces off VWAP (violet
line).

Fig.  11.12:  May 16 2019: Extremely good day. Price lifts over VWAP in a
very clear sense.

Bad days:

Fig.  11.13:  May 21 2019. Day with losses. Price cannot go away from
VWAP (violet line). Few movements.

Mixed day:

Fig.  11.14: May 17 2019: Very bad morning with few movements. Was
stopped out at noon with a short. In the afternoon there is a nice trend, very
simple to trade.

12 RESOURCES

You find my tools and indicators on

https://www.mql5.com/de/users/gronowski/seller#products

The other indicators I use for Metatrader 5 you find on mql5.com (Heiken
Ashi, Resistance&Support, Swing Chart).

The Swiss Backtester EA for backtesting purposes you can get if you send
me an email (terence.gronowski@gmail.com ). Please reference the book.

You find my Youtube Channel by searching after “Swiss Trader”.

13 REFERENCES

Dalton, James, Mind over Markets, Trader’s Press Inc, 1999
Enger, Andre, Gann Swing Chart and hybrid ZigZag implementation, 2016
Metaquotes inc., Metatrader 5, diverse Figures
Slavin Kevin, How Algorithms Shape Our World, Ted Talk 2014
Tradingview.com: Diverse figures
Tradovate.com: Future Platform, diverse Figures

14 APPENDIX

Journal

It is most important to write a journal. I handwrite my journal as it is faster
than typing. I just note the most important properties. In any case you
should note long/short, start and end time and price.

Trading-Form Page. 1  Date  Wed May 5 2019                  Opeinng R. Dax:
no Dax
today
               Opening R. Dow: 26705/26689
Remarks: Consult S10 Chart  EMA200/S10, Rebounds
Fund: …7692.86…….

 1.

    2.
 _______
 _______
 _______
 _______
 _______
 _______
 _______
 _______
 _______

Asset
Signal
Long/short
Price
Start Time
End Time
Contracts
SL
TP
Dist. VWAP  _______
P/L
P/L Day
Funds act.

 _______
 _______
 _______

 3.

    _______
    _______
    _______
    _______
    _______
    _______
    _______
    _______
    _______

    ______
    ______
    ______
    ______
    ______
    ______
    ______
    ______
    ______

    _______

    ______

    _______
    _______
    _______

    ______
    ______
    ______

Tab. 14.1: Example of a trading journal. It is vital to make a note for every
single trade.

