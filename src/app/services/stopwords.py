"""Stop-so'zlar — o'zbek (lotin+kirill), rus, ingliz.

Nima uchun alohida modul: ilgari ro'yxat ``services/search.py`` ichida qo'lda
yozilgan ~90 so'z edi. Qidiruv so'rovini bo'laklarda tozalash, trigram
nomzodlarini kesish va digest map bosqichi — uchtasi ham bir xil ro'yxatga
muhtoj, shuning uchun bitta manba.

**Barcha so'zlar ``textnorm.uz_fold`` fazosida saqlanadi** — ya'ni lowercase,
apostrofsiz, lotin yozuvida. Kirill variant alohida yozilmaydi: ``ва`` →
``va`` fold'dan keyin o'zi lotinga tushadi. Tekshirish ham fold orqali
(:func:`is_stopword`), aks holda ``Va`` yoki ``ва`` ro'yxatga tushmay qoladi.

Manba: Zenodo o'zbek stop-so'z ro'yxatidan tanlangan yuqori chastotali
bo'lak + rus/ingliz klassik ro'yxatlari. Ro'yxat **qisqa ataylab**: haddan
ziyod agressiv stop-so'z filtri "kim qachon keldi" kabi savollarni
ma'nosiz qoldiradi.
"""

from __future__ import annotations

from app.services.textnorm import uz_fold

__all__ = ["STOP_ALL", "STOP_EN", "STOP_RU", "STOP_UZ", "is_stopword", "strip_stopwords"]

# ─── o'zbek (fold fazosida: kirill variantlari ham shu yerga tushadi) ────────
_UZ = """
va yoki lekin ammo biroq ham hamda yana yoq bor edi emas ekan bolib boldi bolgan
bolsa bolmasa bu shu osha u ular biz siz sen men uni ular ularning uning bizning
sizning mening sening ozi ozining har hech qay qaysi qanday qanaqa qancha nechta
necha nima nima uchun kim kimga kimning qachon qayer qayerda qayerga qayerdan
qanday qilib uchun bilan uchun haqida boyicha orqali kabi singari uchun tomon
tomonidan keyin oldin avval hozir endi hali hamma barcha butun bari kop oz biroz
juda eng faqat ancha deyarli taxminan albatta balki ehtimol demak yani masalan
ya'ni shuning uchun shunday shundan buning bunga bundan unga undan ularga
ulardan menga mendan senga sendan bizga bizdan sizga sizdan qilib qilgan qiladi
qilish boladi bolish kerak mumkin zarur lozim shart yaxshi yomon katta kichik
yangi eski birinchi ikkinchi keyingi oldingi
"""

# ─── rus ─────────────────────────────────────────────────────────────────────
_RU = """
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы
по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг
ли если уже или ни быть был него до вас нибудь опять уж вам сказал ведь там
потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была
сам чтоб без будто человек чего раз тоже себе под жизнь будет ж тогда кто этот
говорил того потому этого какой совсем ним здесь этом один почти мой тем чтобы
нее кажется сейчас были куда зачем сказать всех никогда сегодня можно при
наконец два об другой хоть после над больше тот через эти нас про всего них
какая много разве сказала три эту моя впрочем хорошо свою этой перед иногда
лучше чуть том нельзя такой им более всегда конечно всю между
"""

# ─── ingliz ──────────────────────────────────────────────────────────────────
_EN = """
a an the and or but if then else of to in on at for from by with about into over
after before between under above is are was were be been being am do does did
doing have has had having i you he she it we they me him her us them my your his
its our their this that these these those there here what which who whom when
where why how all any both each few more most other some such no nor not only own
same so than too very can will just should now
"""


def _load(raw: str) -> frozenset[str]:
    return frozenset(uz_fold(w) for w in raw.split() if w)


STOP_UZ: frozenset[str] = _load(_UZ)
STOP_RU: frozenset[str] = _load(_RU)
STOP_EN: frozenset[str] = _load(_EN)
STOP_ALL: frozenset[str] = STOP_UZ | STOP_RU | STOP_EN


def is_stopword(word: str) -> bool:
    """So'z stop-ro'yxatdami (yozuv/apostrof/registrga bog'liq emas).

    Args:
        word: bitta so'z (fold qilinmagan bo'lishi mumkin).

    Returns:
        ``True`` — qidiruv termi sifatida tashlab ketilsin.

    Examples:
        >>> is_stopword("Va"), is_stopword("ва"), is_stopword("to'lov")
        (True, True, False)
    """
    return uz_fold(word) in STOP_ALL


def strip_stopwords(words: list[str]) -> list[str]:
    """Stop-so'zlarni olib tashlaydi; hammasi stop bo'lsa — asl ro'yxat.

    Bo'sh ro'yxat qaytarish qidiruvni "hech nima" ga aylantiradi, shuning
    uchun to'liq bo'shalish holatida kirish o'zgarishsiz qaytariladi.
    """
    kept = [w for w in words if not is_stopword(w)]
    return kept or words
