import pandas as pd
import numpy as np
import json
import struct
import os
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
OUT = "/Users/teo/Downloads/datcreativity/analysis_C2"

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("/Users/teo/Downloads/DAT-RU Results - Sheet1 (1).csv")

def parse_words(w_str):
    if pd.isna(w_str):
        return []
    return [w.strip().lower() for w in w_str.split(',')]

df['word_list'] = df['words'].apply(parse_words)

all_words = []
for wl in df['word_list']:
    all_words.extend(wl)
word_counts = Counter(all_words)
top500 = [w for w, c in word_counts.most_common(500)]

# ============================================================
# 2. SEMANTIC CATEGORIZATION (500 words)
# ============================================================
cat_map = {
    # ANIMAL
    'собака': 'ANIMAL', 'кот': 'ANIMAL', 'кошка': 'ANIMAL', 'слон': 'ANIMAL',
    'мышь': 'ANIMAL', 'рыба': 'ANIMAL', 'лошадь': 'ANIMAL', 'корова': 'ANIMAL',
    'крокодил': 'ANIMAL', 'курица': 'ANIMAL', 'крот': 'ANIMAL', 'кит': 'ANIMAL',
    'птица': 'ANIMAL', 'жираф': 'ANIMAL', 'пингвин': 'ANIMAL', 'медведь': 'ANIMAL',
    'динозавр': 'ANIMAL', 'муравей': 'ANIMAL', 'крыса': 'ANIMAL', 'выхухоль': 'ANIMAL',
    'заяц': 'ANIMAL', 'акула': 'ANIMAL', 'лось': 'ANIMAL', 'пес': 'ANIMAL',
    'лев': 'ANIMAL', 'вомбат': 'ANIMAL', 'бактерия': 'ANIMAL',
    'медуза': 'ANIMAL', 'бегемот': 'ANIMAL', 'волк': 'ANIMAL', 'инфузория': 'ANIMAL',
    'амеба': 'ANIMAL', 'кенгуру': 'ANIMAL', 'осьминог': 'ANIMAL', 'дракон': 'ANIMAL',
    'олень': 'ANIMAL', 'обезьяна': 'ANIMAL', 'котенок': 'ANIMAL', 'верблюд': 'ANIMAL',
    'хомяк': 'ANIMAL', 'животное': 'ANIMAL', 'существо': 'ANIMAL',
    'вирус': 'ANIMAL', 'конь': 'ANIMAL',

    # FOOD
    'яблоко': 'FOOD', 'молоко': 'FOOD', 'хлеб': 'FOOD', 'банан': 'FOOD',
    'помидор': 'FOOD', 'огурец': 'FOOD', 'арбуз': 'FOOD', 'морковь': 'FOOD',
    'капуста': 'FOOD', 'гриб': 'FOOD', 'апельсин': 'FOOD', 'суп': 'FOOD',
    'кофе': 'FOOD', 'соль': 'FOOD', 'торт': 'FOOD', 'колбаса': 'FOOD',
    'картошка': 'FOOD', 'каша': 'FOOD', 'макароны': 'FOOD', 'картофель': 'FOOD',
    'лук': 'FOOD', 'яйцо': 'FOOD', 'вино': 'FOOD', 'пирог': 'FOOD',
    'масло': 'FOOD', 'мясо': 'FOOD', 'мороженое': 'FOOD', 'ананас': 'FOOD',
    'лимон': 'FOOD', 'компот': 'FOOD', 'конфета': 'FOOD', 'сахар': 'FOOD',
    'булка': 'FOOD', 'сок': 'FOOD', 'сметана': 'FOOD', 'шоколад': 'FOOD',
    'блин': 'FOOD', 'перец': 'FOOD', 'пиво': 'FOOD', 'чай': 'FOOD',
    'сыр': 'FOOD', 'еда': 'FOOD', 'груша': 'FOOD', 'хрен': 'FOOD',

    # BODY
    'рука': 'BODY', 'нога': 'BODY', 'глаз': 'BODY', 'нос': 'BODY',
    'зуб': 'BODY', 'кровь': 'BODY', 'мозг': 'BODY', 'волос': 'BODY',
    'ноготь': 'BODY', 'палец': 'BODY', 'ухо': 'BODY', 'сердце': 'BODY',
    'кожа': 'BODY', 'голова': 'BODY', 'кость': 'BODY', 'печень': 'BODY',
    'язык': 'BODY', 'почка': 'BODY', 'прыщ': 'BODY', 'сопля': 'BODY',
    'жопа': 'BODY', 'член': 'BODY', 'пенис': 'BODY',
    'какашка': 'BODY', 'говно': 'BODY', 'хуй': 'BODY',

    # NATURE
    'небо': 'NATURE', 'солнце': 'NATURE', 'дерево': 'NATURE', 'море': 'NATURE',
    'снег': 'NATURE', 'облако': 'NATURE', 'вода': 'NATURE', 'луна': 'NATURE',
    'звезда': 'NATURE', 'река': 'NATURE', 'ветер': 'NATURE', 'камень': 'NATURE',
    'цветок': 'NATURE', 'трава': 'NATURE', 'гора': 'NATURE', 'туман': 'NATURE',
    'земля': 'NATURE', 'дождь': 'NATURE', 'огонь': 'NATURE', 'песок': 'NATURE',
    'лист': 'NATURE', 'радуга': 'NATURE', 'комета': 'NATURE', 'озеро': 'NATURE',
    'кактус': 'NATURE', 'воздух': 'NATURE', 'лед': 'NATURE', 'мороз': 'NATURE',
    'весна': 'NATURE', 'роза': 'NATURE', 'корень': 'NATURE', 'вулкан': 'NATURE',
    'растение': 'NATURE', 'лужа': 'NATURE', 'скала': 'NATURE', 'елка': 'NATURE',
    'жимолость': 'NATURE', 'береза': 'NATURE', 'верба': 'NATURE', 'лава': 'NATURE',
    'магма': 'NATURE', 'айсберг': 'NATURE', 'атмосфера': 'NATURE',
    'стратосфера': 'NATURE', 'грязь': 'NATURE', 'пыль': 'NATURE',
    'зима': 'NATURE', 'лето': 'NATURE', 'ноябрь': 'NATURE', 'погода': 'NATURE',
    'свет': 'NATURE', 'ночь': 'NATURE', 'день': 'NATURE', 'нефть': 'NATURE',
    'золото': 'NATURE', 'железо': 'NATURE', 'сталь': 'NATURE', 'пустыня': 'NATURE',
    'синева': 'NATURE', 'сияние': 'NATURE', 'шерсть': 'NATURE',

    # OBJECT_HOME
    'стол': 'OBJECT_HOME', 'стул': 'OBJECT_HOME', 'кровать': 'OBJECT_HOME',
    'диван': 'OBJECT_HOME', 'ложка': 'OBJECT_HOME', 'шкаф': 'OBJECT_HOME',
    'окно': 'OBJECT_HOME', 'дверь': 'OBJECT_HOME', 'подушка': 'OBJECT_HOME',
    'ковер': 'OBJECT_HOME', 'тарелка': 'OBJECT_HOME', 'вилка': 'OBJECT_HOME',
    'одеяло': 'OBJECT_HOME', 'стена': 'OBJECT_HOME', 'бумага': 'OBJECT_HOME',
    'бутылка': 'OBJECT_HOME', 'очки': 'OBJECT_HOME', 'кружка': 'OBJECT_HOME',
    'ведро': 'OBJECT_HOME', 'лопата': 'OBJECT_HOME', 'стекло': 'OBJECT_HOME',
    'тетрадь': 'OBJECT_HOME', 'нож': 'OBJECT_HOME', 'коробка': 'OBJECT_HOME',
    'полотенце': 'OBJECT_HOME', 'зеркало': 'OBJECT_HOME', 'штора': 'OBJECT_HOME',
    'люстра': 'OBJECT_HOME', 'чашка': 'OBJECT_HOME', 'краска': 'OBJECT_HOME',
    'стакан': 'OBJECT_HOME', 'кастрюля': 'OBJECT_HOME', 'топор': 'OBJECT_HOME',
    'кресло': 'OBJECT_HOME', 'доска': 'OBJECT_HOME', 'забор': 'OBJECT_HOME',
    'ножницы': 'OBJECT_HOME', 'труба': 'OBJECT_HOME', 'лестница': 'OBJECT_HOME',
    'пакет': 'OBJECT_HOME', 'кольцо': 'OBJECT_HOME', 'перо': 'OBJECT_HOME',
    'свеча': 'OBJECT_HOME', 'зонт': 'OBJECT_HOME', 'ключ': 'OBJECT_HOME',
    'якорь': 'OBJECT_HOME', 'отвертка': 'OBJECT_HOME', 'стамеска': 'OBJECT_HOME',
    'картон': 'OBJECT_HOME', 'лом': 'OBJECT_HOME', 'простыня': 'OBJECT_HOME',
    'чемодан': 'OBJECT_HOME', 'обои': 'OBJECT_HOME', 'салфетка': 'OBJECT_HOME',
    'тряпка': 'OBJECT_HOME', 'полка': 'OBJECT_HOME', 'табурет': 'OBJECT_HOME',
    'табуретка': 'OBJECT_HOME', 'сумка': 'OBJECT_HOME', 'гвоздь': 'OBJECT_HOME',
    'тапка': 'OBJECT_HOME', 'сапог': 'OBJECT_HOME', 'потолок': 'OBJECT_HOME',
    'клей': 'OBJECT_HOME', 'крыша': 'OBJECT_HOME', 'мост': 'OBJECT_HOME',
    'бетон': 'OBJECT_HOME', 'столб': 'OBJECT_HOME', 'асфальт': 'OBJECT_HOME',
    'шуруп': 'OBJECT_HOME', 'лампа': 'OBJECT_HOME', 'мяч': 'OBJECT_HOME',
    'кирпич': 'OBJECT_HOME', 'носок': 'OBJECT_HOME', 'пол': 'OBJECT_HOME',
    'колесо': 'OBJECT_HOME', 'башня': 'OBJECT_HOME', 'мыло': 'OBJECT_HOME',
    'часы': 'OBJECT_HOME', 'дыра': 'OBJECT_HOME', 'палка': 'OBJECT_HOME',
    'мусор': 'OBJECT_HOME', 'рюкзак': 'OBJECT_HOME', 'пылесос': 'OBJECT_HOME',
    'кран': 'OBJECT_HOME', 'игла': 'OBJECT_HOME', 'холодильник': 'OBJECT_HOME',
    'батарея': 'OBJECT_HOME', 'утюг': 'OBJECT_HOME', 'чайник': 'OBJECT_HOME',
    'унитаз': 'OBJECT_HOME', 'пистолет': 'OBJECT_HOME', 'замок': 'OBJECT_HOME',
    'гараж': 'OBJECT_HOME', 'кнопка': 'OBJECT_HOME',
    'игрушка': 'OBJECT_HOME', 'кукла': 'OBJECT_HOME',
    'ручка': 'OBJECT_HOME', 'карандаш': 'OBJECT_HOME',
    'молоток': 'OBJECT_HOME',
    'платье': 'OBJECT_HOME', 'штаны': 'OBJECT_HOME', 'куртка': 'OBJECT_HOME',
    'пиджак': 'OBJECT_HOME', 'шапка': 'OBJECT_HOME', 'кофта': 'OBJECT_HOME',
    'рубашка': 'OBJECT_HOME', 'юбка': 'OBJECT_HOME',
    'ботинок': 'OBJECT_HOME', 'фонарь': 'OBJECT_HOME',
    'шар': 'OBJECT_HOME', 'циркуль': 'OBJECT_HOME',
    'сигарета': 'OBJECT_HOME', 'заноза': 'OBJECT_HOME',
    'коромысло': 'OBJECT_HOME', 'бигель': 'OBJECT_HOME',
    'шляпа': 'OBJECT_HOME',

    # OBJECT_TECH
    'машина': 'OBJECT_TECH', 'самолет': 'OBJECT_TECH', 'телефон': 'OBJECT_TECH',
    'автомобиль': 'OBJECT_TECH', 'компьютер': 'OBJECT_TECH', 'ракета': 'OBJECT_TECH',
    'телевизор': 'OBJECT_TECH', 'велосипед': 'OBJECT_TECH', 'клавиатура': 'OBJECT_TECH',
    'трактор': 'OBJECT_TECH', 'поезд': 'OBJECT_TECH', 'автобус': 'OBJECT_TECH',
    'корабль': 'OBJECT_TECH', 'робот': 'OBJECT_TECH', 'ноутбук': 'OBJECT_TECH',
    'танк': 'OBJECT_TECH', 'паровоз': 'OBJECT_TECH', 'вертолет': 'OBJECT_TECH',
    'планшет': 'OBJECT_TECH', 'наушник': 'OBJECT_TECH', 'двигатель': 'OBJECT_TECH',
    'синхрофазотрон': 'OBJECT_TECH', 'карбюратор': 'OBJECT_TECH', 'трамвай': 'OBJECT_TECH',
    'метро': 'OBJECT_TECH', 'парашют': 'OBJECT_TECH', 'дирижабль': 'OBJECT_TECH',
    'лодка': 'OBJECT_TECH', 'лыжа': 'OBJECT_TECH', 'капот': 'OBJECT_TECH',
    'втулка': 'OBJECT_TECH',

    # PLACE
    'дом': 'PLACE', 'космос': 'PLACE', 'лес': 'PLACE', 'океан': 'PLACE',
    'планета': 'PLACE', 'вселенная': 'PLACE', 'школа': 'PLACE', 'город': 'PLACE',
    'дорога': 'PLACE', 'поле': 'PLACE', 'галактика': 'PLACE', 'страна': 'PLACE',
    'улица': 'PLACE', 'остров': 'PLACE', 'туалет': 'PLACE', 'магазин': 'PLACE',
    'квартира': 'PLACE', 'здание': 'PLACE', 'библиотека': 'PLACE', 'аптека': 'PLACE',
    'юпитер': 'PLACE', 'марс': 'PLACE', 'мир': 'PLACE',
    'церковь': 'PLACE', 'театр': 'PLACE', 'вытрезвитель': 'PLACE',

    # ABSTRACT
    'любовь': 'ABSTRACT', 'жизнь': 'ABSTRACT', 'сон': 'ABSTRACT',
    'красота': 'ABSTRACT', 'смысл': 'ABSTRACT', 'смерть': 'ABSTRACT',
    'философия': 'ABSTRACT', 'абстракция': 'ABSTRACT', 'сингулярность': 'ABSTRACT',
    'чувственность': 'ABSTRACT', 'история': 'ABSTRACT', 'мысль': 'ABSTRACT',
    'момент': 'ABSTRACT', 'скорость': 'ABSTRACT', 'время': 'ABSTRACT',
    'амбивалентность': 'ABSTRACT', 'путешествие': 'ABSTRACT', 'мечта': 'ABSTRACT',
    'совесть': 'ABSTRACT', 'душа': 'ABSTRACT', 'правда': 'ABSTRACT',
    'сила': 'ABSTRACT', 'болезнь': 'ABSTRACT', 'сущность': 'ABSTRACT',
    'трансцендентность': 'ABSTRACT', 'дух': 'ABSTRACT',
    'креативность': 'ABSTRACT', 'креатив': 'ABSTRACT',
    'волюнтаризм': 'ABSTRACT', 'клаустрофобия': 'ABSTRACT', 'шизофрения': 'ABSTRACT',
    'религия': 'ABSTRACT', 'культ': 'ABSTRACT', 'бог': 'ABSTRACT',
    'война': 'ABSTRACT', 'секс': 'ABSTRACT', 'счастие': 'ABSTRACT',
    'идиот': 'ABSTRACT', 'прелесть': 'ABSTRACT', 'нежность': 'ABSTRACT',
    'конституция': 'ABSTRACT', 'революция': 'ABSTRACT',
    'деньга': 'ABSTRACT', 'слово': 'ABSTRACT', 'существительное': 'ABSTRACT',
    'глагол': 'ABSTRACT', 'буква': 'ABSTRACT',
    'полиция': 'ABSTRACT', 'цвет': 'ABSTRACT', 'пауза': 'ABSTRACT',
    'якудз': 'ABSTRACT', 'человеконенавистничество': 'ABSTRACT',
    'одина': 'ABSTRACT', 'трей': 'ABSTRACT', 'потуга': 'ABSTRACT',
    'двом': 'ABSTRACT', 'клич': 'ABSTRACT', 'свобода': 'ABSTRACT',

    # SCIENCE
    'математика': 'SCIENCE', 'атом': 'SCIENCE', 'молекула': 'SCIENCE',
    'алгоритм': 'SCIENCE', 'квазар': 'SCIENCE', 'кварк': 'SCIENCE',
    'интеграл': 'SCIENCE', 'астероид': 'SCIENCE', 'физика': 'SCIENCE',
    'квант': 'SCIENCE', 'ядро': 'SCIENCE', 'электричество': 'SCIENCE',
    'митохондрия': 'SCIENCE', 'квадрат': 'SCIENCE', 'орбита': 'SCIENCE',
    'электрон': 'SCIENCE', 'косинус': 'SCIENCE', 'синус': 'SCIENCE',
    'энтропия': 'SCIENCE', 'треугольник': 'SCIENCE', 'гравитация': 'SCIENCE',
    'протон': 'SCIENCE', 'кислород': 'SCIENCE', 'кислота': 'SCIENCE',
    'нейрон': 'SCIENCE', 'валентность': 'SCIENCE', 'гипотенуза': 'SCIENCE',
    'гипербола': 'SCIENCE', 'параллелепипед': 'SCIENCE', 'дивергенция': 'SCIENCE',
    'химия': 'SCIENCE', 'волна': 'SCIENCE', 'круг': 'SCIENCE',
    'цифра': 'SCIENCE', 'таблетка': 'SCIENCE', 'лекарство': 'SCIENCE',

    # PERSON
    'мама': 'PERSON', 'человек': 'PERSON', 'ребенок': 'PERSON',
    'мать': 'PERSON', 'учитель': 'PERSON', 'папа': 'PERSON',
    'друг': 'PERSON', 'врач': 'PERSON', 'сын': 'PERSON',
    'семья': 'PERSON', 'принцесса': 'PERSON', 'президент': 'PERSON',
    'гном': 'PERSON', 'трус': 'PERSON', 'лакей': 'PERSON',
    'путина': 'PERSON',

    # ART_CULTURE
    'книга': 'ART_CULTURE', 'картина': 'ART_CULTURE', 'музыка': 'ART_CULTURE',
    'словарь': 'ART_CULTURE', 'песня': 'ART_CULTURE', 'кино': 'ART_CULTURE',
    'гитара': 'ART_CULTURE', 'пианино': 'ART_CULTURE', 'скрипка': 'ART_CULTURE',
    'сказка': 'ART_CULTURE', 'искусство': 'ART_CULTURE', 'танец': 'ART_CULTURE',
    'икона': 'ART_CULTURE', 'литература': 'ART_CULTURE', 'учебник': 'ART_CULTURE',
    'нота': 'ART_CULTURE', 'пюпитр': 'ART_CULTURE', 'глобус': 'ART_CULTURE',
    'календарь': 'ART_CULTURE', 'карта': 'ART_CULTURE',
    'сольфеджио': 'ART_CULTURE', 'ямб': 'ART_CULTURE',
    'фильм': 'ART_CULTURE',

    # EMOTION
    'радость': 'EMOTION', 'боль': 'EMOTION', 'страх': 'EMOTION',
    'грусть': 'EMOTION', 'зависть': 'EMOTION', 'смех': 'EMOTION',
    'крик': 'EMOTION', 'улыбка': 'EMOTION',

    # ACTION_PROCESS
    'обсуждение': 'ACTION_PROCESS', 'игра': 'ACTION_PROCESS',
    'работа': 'ACTION_PROCESS', 'полет': 'ACTION_PROCESS',
    'бег': 'ACTION_PROCESS', 'исследование': 'ACTION_PROCESS',
    'тест': 'ACTION_PROCESS', 'взрыв': 'ACTION_PROCESS',
    'урок': 'ACTION_PROCESS', 'спорт': 'ACTION_PROCESS',
    'футбол': 'ACTION_PROCESS', 'жест': 'ACTION_PROCESS',
}

# Verify coverage
top500_set = set(top500)
categorized = set(cat_map.keys()) & top500_set
uncategorized = top500_set - set(cat_map.keys())
print(f"Categorized: {len(categorized)}/500")
if uncategorized:
    print(f"Uncategorized ({len(uncategorized)}): {uncategorized}")

# ============================================================
# 3. COMPUTE CATEGORY DISTRIBUTIONS PER SUBMISSION
# ============================================================
def get_submission_categories(word_list):
    cats = []
    for w in word_list:
        if w in cat_map:
            cats.append(cat_map[w])
    return cats

df['categories'] = df['word_list'].apply(get_submission_categories)
df['n_categorized'] = df['categories'].apply(len)
df['unique_cats'] = df['categories'].apply(lambda x: len(set(x)))
df['cat_set'] = df['categories'].apply(lambda x: frozenset(x))

df_filt = df[df['n_categorized'] >= 5].copy()
print(f"\nSubmissions with >=5 categorized words: {len(df_filt)}/{len(df)}")
print(f"Score stats (filtered): mean={df_filt['score'].mean():.2f}, std={df_filt['score'].std():.2f}")

# ============================================================
# 4a. Score ~ number of unique categories
# ============================================================
cat_score = df_filt.groupby('unique_cats')['score'].agg(['mean', 'std', 'count']).reset_index()
cat_score.columns = ['n_unique_cats', 'mean_score', 'std_score', 'count']
print("\n=== Score by number of unique categories ===")
print(cat_score.to_string(index=False))

corr, pval = stats.pearsonr(df_filt['unique_cats'], df_filt['score'])
print(f"\nPearson r = {corr:.4f}, p = {pval:.2e}")
spearman_r, spearman_p = stats.spearmanr(df_filt['unique_cats'], df_filt['score'])
print(f"Spearman rho = {spearman_r:.4f}, p = {spearman_p:.2e}")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(cat_score['n_unique_cats'], cat_score['mean_score'],
              yerr=cat_score['std_score']/np.sqrt(cat_score['count']),
              color='steelblue', edgecolor='black', capsize=4)
for bar, cnt in zip(bars, cat_score['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'n={cnt}', ha='center', va='bottom', fontsize=8)
ax.set_xlabel('Number of Unique Semantic Categories (out of 7 words)')
ax.set_ylabel('Mean DAT Score')
ax.set_title(f'Score vs Category Diversity\n(Pearson r={corr:.3f}, p={pval:.2e})')
ax.set_xticks(cat_score['n_unique_cats'])
plt.tight_layout()
plt.savefig(f'{OUT}/4a_score_vs_category_diversity.png', dpi=150)
plt.close()

# ============================================================
# 4b. Category frequency
# ============================================================
all_cats_flat = []
for cats in df_filt['categories']:
    all_cats_flat.extend(cats)
cat_freq = Counter(all_cats_flat)
cat_freq_df = pd.DataFrame(cat_freq.items(), columns=['category', 'count']).sort_values('count', ascending=False)
print("\n=== Category Frequency ===")
print(cat_freq_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette('Set2', len(cat_freq_df))
ax.bar(cat_freq_df['category'], cat_freq_df['count'], color=colors, edgecolor='black')
ax.set_xlabel('Semantic Category')
ax.set_ylabel('Total Word Count (in filtered submissions)')
ax.set_title('Semantic Category Frequency in DAT Submissions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{OUT}/4b_category_frequency.png', dpi=150)
plt.close()

# ============================================================
# 4c. Category x score
# ============================================================
cat_score_map = defaultdict(list)
for _, row in df_filt.iterrows():
    for cat in set(row['categories']):
        cat_score_map[cat].append(row['score'])

cat_score_df = pd.DataFrame([
    {'category': cat, 'mean_score': np.mean(scores), 'std_score': np.std(scores),
     'n': len(scores), 'sem': np.std(scores)/np.sqrt(len(scores))}
    for cat, scores in cat_score_map.items()
]).sort_values('mean_score', ascending=False)
print("\n=== Mean Score per Category ===")
print(cat_score_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
colors_c = ['#e74c3c' if s > cat_score_df['mean_score'].median() else '#3498db'
            for s in cat_score_df['mean_score']]
ax.bar(cat_score_df['category'], cat_score_df['mean_score'],
       yerr=cat_score_df['sem'], color=colors_c, edgecolor='black', capsize=3)
ax.axhline(df_filt['score'].mean(), color='gray', ls='--', label='Overall mean')
ax.set_xlabel('Semantic Category')
ax.set_ylabel('Mean DAT Score')
ax.set_title('Mean Score by Semantic Category\n(red = above median, blue = below)')
plt.xticks(rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/4c_category_score.png', dpi=150)
plt.close()

# ============================================================
# 4d. Category pair distances from embeddings
# ============================================================
print("\n=== Loading embeddings ===")
with open("/Users/teo/Downloads/datcreativity/data/words.json", "r") as f:
    emb_words = json.load(f)
word2idx = {w: i for i, w in enumerate(emb_words)}

with open("/Users/teo/Downloads/datcreativity/data/matrix.bin", "rb") as f:
    n_words_emb, dim = struct.unpack('II', f.read(8))
    raw = np.frombuffer(f.read(), dtype=np.int8).reshape(n_words_emb, dim)
    embeddings = raw.astype(np.float32)

print(f"Embeddings: {n_words_emb} x {dim}")

cat_embeddings = defaultdict(list)
for word in top500:
    if word in cat_map and word in word2idx:
        cat = cat_map[word]
        cat_embeddings[cat].append(embeddings[word2idx[word]])

cat_names = sorted(cat_embeddings.keys())
cat_means = {}
for cat in cat_names:
    vecs = np.array(cat_embeddings[cat])
    cat_means[cat] = vecs.mean(axis=0)

n_cats = len(cat_names)
dist_matrix = np.zeros((n_cats, n_cats))
for i in range(n_cats):
    for j in range(n_cats):
        v1, v2 = cat_means[cat_names[i]], cat_means[cat_names[j]]
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        dist_matrix[i, j] = 1 - cos_sim

pair_dists = {}
for i in range(n_cats):
    for j in range(i+1, n_cats):
        vecs_i = np.array(cat_embeddings[cat_names[i]])
        vecs_j = np.array(cat_embeddings[cat_names[j]])
        norms_i = np.linalg.norm(vecs_i, axis=1, keepdims=True) + 1e-10
        norms_j = np.linalg.norm(vecs_j, axis=1, keepdims=True) + 1e-10
        cos_sim = (vecs_i / norms_i) @ (vecs_j / norms_j).T
        mean_dist = np.mean(1 - cos_sim)
        pair_dists[(cat_names[i], cat_names[j])] = mean_dist

sorted_pairs = sorted(pair_dists.items(), key=lambda x: x[1], reverse=True)
print("\n=== Top 10 Most Distant Category Pairs ===")
for (c1, c2), d in sorted_pairs[:10]:
    print(f"  {c1} <-> {c2}: {d:.4f}")
print("\n=== Top 10 Closest Category Pairs ===")
for (c1, c2), d in sorted_pairs[-10:]:
    print(f"  {c1} <-> {c2}: {d:.4f}")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(dist_matrix, xticklabels=cat_names, yticklabels=cat_names,
            annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax, vmin=0)
ax.set_title('Cosine Distance Between Category Centroids')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{OUT}/4d_category_pair_distances.png', dpi=150)
plt.close()

# ============================================================
# 4e. High vs low performers
# ============================================================
q90 = df_filt['score'].quantile(0.90)
q10 = df_filt['score'].quantile(0.10)
df_top = df_filt[df_filt['score'] >= q90]
df_bot = df_filt[df_filt['score'] <= q10]
print(f"\nTop 10%: n={len(df_top)}, score >= {q90:.2f}")
print(f"Bottom 10%: n={len(df_bot)}, score <= {q10:.2f}")

def get_cat_dist(subset):
    all_c = []
    for cats in subset['categories']:
        all_c.extend(cats)
    total = len(all_c)
    freq = Counter(all_c)
    return {k: v/total for k, v in freq.items()}

top_dist = get_cat_dist(df_top)
bot_dist = get_cat_dist(df_bot)

all_category_names = sorted(set(list(top_dist.keys()) + list(bot_dist.keys())))
comparison = pd.DataFrame({
    'category': all_category_names,
    'top10pct': [top_dist.get(c, 0) for c in all_category_names],
    'bot10pct': [bot_dist.get(c, 0) for c in all_category_names],
})
comparison['diff'] = comparison['top10pct'] - comparison['bot10pct']
comparison = comparison.sort_values('diff', ascending=False)
print("\n=== Category Distribution: Top 10% vs Bottom 10% ===")
print(comparison.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison))
width = 0.35
ax.bar(x - width/2, comparison['top10pct'], width, label='Top 10%', color='#e74c3c', edgecolor='black')
ax.bar(x + width/2, comparison['bot10pct'], width, label='Bottom 10%', color='#3498db', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(comparison['category'], rotation=45, ha='right')
ax.set_ylabel('Proportion of Words')
ax.set_title('Category Distribution: Top 10% vs Bottom 10% Scorers')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/4e_high_vs_low_categories.png', dpi=150)
plt.close()

# ============================================================
# 4f. Optimal category recipe
# ============================================================
combo_scores = defaultdict(list)
for _, row in df_filt.iterrows():
    key = tuple(sorted(set(row['categories'])))
    combo_scores[key].append(row['score'])

combo_df = pd.DataFrame([
    {'combination': ' + '.join(k), 'n_cats': len(k), 'mean_score': np.mean(v),
     'std_score': np.std(v), 'count': len(v)}
    for k, v in combo_scores.items()
]).sort_values('mean_score', ascending=False)

combo_df_stable = combo_df[combo_df['count'] >= 10].copy()
print("\n=== Top 15 Category Combinations (n>=10) by Mean Score ===")
print(combo_df_stable.head(15).to_string(index=False))
print("\n=== Bottom 10 Category Combinations (n>=10) by Mean Score ===")
print(combo_df_stable.tail(10).to_string(index=False))

for nc in range(3, 8):
    sub = combo_df_stable[combo_df_stable['n_cats'] == nc]
    if len(sub) > 0:
        best = sub.iloc[0]
        print(f"\nBest {nc}-category combo: {best['combination']} (score={best['mean_score']:.2f}, n={best['count']})")

# ============================================================
# 5. TABLES
# ============================================================
print("\n" + "="*60)
print("=== TOP 10 WORDS PER CATEGORY ===")
print("="*60)

top500_freq = {w: word_counts[w] for w in top500}

tables_data = {}
for cat in sorted(set(cat_map.values())):
    cat_words = [(w, top500_freq.get(w, 0)) for w in top500 if w in cat_map and cat_map[w] == cat]
    cat_words.sort(key=lambda x: x[1], reverse=True)
    tables_data[cat] = cat_words[:10]
    print(f"\n{cat}:")
    for w, c in cat_words[:10]:
        print(f"  {w}: {c}")

# Mean score contribution per category
print("\n=== Mean Score Contribution per Category ===")
print(cat_score_df[['category', 'mean_score', 'n']].to_string(index=False))

# Save summary
with open(f'{OUT}/summary_results.txt', 'w') as f:
    f.write("DAT-RU Semantic Category Analysis (C2)\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total submissions: {len(df)}\n")
    f.write(f"Submissions with >=5 categorized words: {len(df_filt)}\n")
    f.write(f"Top 500 words categorized: {len(categorized)}/500\n\n")

    f.write("4a. Score ~ Category Diversity\n")
    f.write(f"Pearson r = {corr:.4f}, p = {pval:.2e}\n")
    f.write(f"Spearman rho = {spearman_r:.4f}, p = {spearman_p:.2e}\n")
    f.write(cat_score.to_string(index=False) + "\n\n")

    f.write("4b. Category Frequency\n")
    f.write(cat_freq_df.to_string(index=False) + "\n\n")

    f.write("4c. Mean Score per Category\n")
    f.write(cat_score_df.to_string(index=False) + "\n\n")

    f.write("4d. Most Distant Category Pairs\n")
    for (c1, c2), d in sorted_pairs[:10]:
        f.write(f"  {c1} <-> {c2}: {d:.4f}\n")
    f.write("\nClosest Category Pairs\n")
    for (c1, c2), d in sorted_pairs[-10:]:
        f.write(f"  {c1} <-> {c2}: {d:.4f}\n")

    f.write("\n4e. Category Distribution: Top 10% vs Bottom 10%\n")
    f.write(comparison.to_string(index=False) + "\n\n")

    f.write("4f. Top 15 Category Combinations (n>=10)\n")
    f.write(combo_df_stable.head(15).to_string(index=False) + "\n\n")
    f.write("Bottom 10 Category Combinations (n>=10)\n")
    f.write(combo_df_stable.tail(10).to_string(index=False) + "\n\n")

    f.write("5. Top 10 Words per Category\n")
    for cat in sorted(tables_data.keys()):
        f.write(f"\n{cat}:\n")
        for w, c in tables_data[cat]:
            f.write(f"  {w}: {c}\n")

with open(f'{OUT}/category_map.json', 'w') as f:
    json.dump(cat_map, f, ensure_ascii=False, indent=2)

print("\n\nAll plots and results saved to", OUT)
print("DONE")
