# 07. Источники

Откуда взято каждое правило и что источник на самом деле утверждает.
Если реализация расходится с источником — записать в JOURNAL.

## Тренд и моментум

| Источник | Что утверждает | Где используется |
|---|---|---|
| Zarattini, Pagani, Barbon (2025). *Catching Crypto Trends; A Tactical Approach for Bitcoin and Altcoins.* SSRN 5209907 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907 | Ансамбль Donchian-моделей с разными окнами + размер по волатильности на ротационном портфеле топ-20 ликвидных монет, 2015–03.2025, без ошибки выжившего: Шарп > 1.5 после комиссий. Только лонг | S01, S01b, O2 |
| CXO Advisory, пересказ правил той же статьи — https://www.cxoadvisory.com/technical-trading/crypto-asset-trend-following-strategies/ | Окна, трейлинг по середине канала, 25% целевая волатильность, лонг-онли | S01 (сверить с PDF) |
| Liu, Tsyvinski (2021). *Risks and Returns of Cryptocurrency.* RFS — https://www.nber.org/system/files/working_papers/w24877/w24877.pdf | Time-series моментум на горизонтах 1–4 недели; доходность рынка предсказывает будущую до 8 недель | S02 |
| Liu, Tsyvinski, Wu (2022). *Common Risk Factors in Cryptocurrency.* Journal of Finance — https://ssrn.com/abstract=3379131 | Кросс-секционный моментум с окном формирования 1–4 недели, ~3% в неделю long/short на широкой вселенной | S13 |
| *Cryptocurrency momentum has (not) its moments* (2025), FMPM — https://link.springer.com/article/10.1007/s11408-025-00474-9 | Крипто-моментум подвержен обвалам; управление волатильностью их смягчает; эффект связан с крупными монетами | S13, O2 |
| Hudson, Urquhart (2021). *Technical trading and cryptocurrencies.* Annals of OR — https://centaur.reading.ac.uk/85715/8/Hudson-Urquhart2019_Article_TechnicalTradingAndCryptocurre.pdf | ~15 000 технических правил сохраняют значимость после поправок на data-snooping, снижают просадки; вне выборки нет предсказуемости для BTC, но есть для других монет | S03–S05 (обоснование класса) |
| Grobys, Ahmed, Sapkota (2020). *Technical trading rules in the cryptocurrency market.* FRL — https://www.sciencedirect.com/science/article/pii/S1544612319308852 | 20-дневная средняя на 11 монетах (2016–2018): без BTC ~8.76% годовых сверх рынка | S03 |
| *Are simple technical trading rules profitable in bitcoin markets?* (2024), IREF — https://www.sciencedirect.com/science/article/abs/pii/S1059056024003010 | Дневная торговля устойчивее к издержкам, чем внутридневная | Выбор ТФ |
| Arda (2025). *Bollinger Bands under Varying Market Regimes.* SSRN 5775962 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5775962 | На BTC/USDT в медвежьей фазе пробой Боллинджера лучше, возврат к среднему проваливается | S05, S11 |

## Возврат к среднему и внутридневная предсказуемость

| Источник | Что утверждает | Где используется |
|---|---|---|
| Zaremba и др. (2021). *Up or down? Short-term reversal, momentum, and liquidity effects in cryptocurrency markets.* IRFA — https://www.sciencedirect.com/science/article/pii/S1057521921002349 | Дневной разворот вызван неликвидностью; у крупнейших монет — дневной моментум, а не разворот | Ожидания для S10, S11 |
| Wen, Bouri, Xu, Zhao (2022). *Intraday return predictability in the cryptocurrency markets.* NAJEF — https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833 | Внутридневные моментум и разворот зависят от скачков, ликвидности, новостей | Контекст для 1h |
| Baquero (2026). *Bitcoin Price Prediction: Peer-Reviewed Evidence and Social Media Discourse.* arXiv 2606.00071 — https://arxiv.org/pdf/2606.00071 | На часовом ТФ модели не обгоняют случайное блуждание, на дневном обгоняют; статистическая значимость ≠ прибыль после издержек; методические требования (walk-forward, multi-regime holdout, наивный бенчмарк) | Протокол, выбор ТФ |
| Coinquant (2026). *Mean-reversion in crypto: evidence from 78 backtests* — https://www.coinquant.ai/blog/building-a-mean-reversion-strategy-in-cryptocurrency-markets-evidence-from-78-backtests | Результат mean reversion определяется режимом рынка; быстрые ТФ съедаются издержками | S10, S11 (ожидания) |
| Connors, Alvarez (2008). *Short Term Trading Strategies That Work* (книга) | RSI(2) с фильтром SMA200, выход по SMA5 | S10 |

## Крипто-специфика

| Источник | Что утверждает | Где используется |
|---|---|---|
| Schmeling, Schrimpf, Todorov (2023). *Crypto carry.* BIS WP 1087 — https://www.bis.org/publ/work1087.pdf | Carry фьючерсов бывает огромным; высокий carry предсказывает будущие обвалы; драйвер — погоня за трендом мелких инвесторов с плечом | S12 |
| Guo, Sang, Tu, Wang (2024). *Cross-cryptocurrency return predictability.* JEDC — https://research-information.bris.ac.uk/en/publications/cross-cryptocurrency-return-predictability/ | Лаговые доходности других монет предсказывают доходность целевой (данные Binance) | O1 (косвенно) |

## Популярные стратегии (практика трейдеров)

| Стратегия | Источник | Где |
|---|---|---|
| Turtle System 1 | Curtis Faith, *Way of the Turtle* (2007); правила Turtle Traders | S04 |
| Supertrend | Olivier Seban; стандартная реализация TradingView | S06 |
| Golden / death cross | Классический технический анализ | S07 |
| MACD | Gerald Appel | S08 |
| Volatility breakout | Larry Williams, *Long-Term Secrets to Short-Term Trading* | S09 |

## Методология оценки

| Источник | Где используется |
|---|---|
| Bailey, Borwein, López de Prado, Zhu (2014). *Pseudo-Mathematics and Financial Charlatanism.* Notices of the AMS — https://www.ams.org/notices/201405/rnoti-p458.pdf | Переобучение бэктестов, PBO/CSCV |
| Bailey, López de Prado (2014). *The Deflated Sharpe Ratio.* Journal of Portfolio Management | DSR в метриках и лидерборде |

## Рассмотрено и отклонено

| Идея | Почему не берём |
|---|---|
| Сеточные боты | Нужны лимитные ордера (в песочнице пока только рыночные); в обвал сетка на лонг набирает убыточную позицию |
| Паттерны, Ишимоку | Много параметров, нет устойчивой доказательной базы на крипте |
| Сезонность по часам суток | Работа на BTC/ETH 2016–2025 (MDPI JRFM, https://www.mdpi.com/1911-8074/19/9/692): найденные правила не прошли бутстрэп и тест Хансена SPA |
| Лаг «BTC → альты» как самостоятельная стратегия | Задержка реакции найдена у мелких монет на минутных данных (Springer APFM 2026, https://link.springer.com/article/10.1007/s10690-026-09589-z) — не наш ТФ и не наши монеты |
| ML-модели | Отдельный проект; здесь — только правила без обучения |
