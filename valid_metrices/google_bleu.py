# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.
This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math
import math
import sys
import fractions
import warnings
from collections import Counter
try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction



from nltk.util import ngrams

def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts



def corpus_bleu(hypotheses, references):
    refs = []
    hyps = []
    count = 0
    total_score = 0.0

    assert (sorted(hypotheses.keys()) == sorted(references.keys()))
    Ids = list(hypotheses.keys())
    ind_score = dict()

    for id in Ids:
        hyp = hypotheses[id][0].split()
        ref = [r.split() for r in references[id]]
        hyps.append(hyp)
        refs.append(ref)

        score = compute_bleu([ref], [hyp], smooth=True)[0]
        total_score += score
        count += 1
        ind_score[id] = score

    avg_score = total_score / count
    corpus_bleu = compute_bleu(refs, hyps, smooth=True)[0]
    return corpus_bleu, avg_score, ind_score
def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)

def sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25),
                  smoothing_function=None, auto_reweigh=False):
    return corpus_bleu1([references], [hypothesis],
                       weights, smoothing_function, auto_reweigh)


def corpus_bleu1(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=None, auto_reweigh=False):
    p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0
    assert len(list_of_references) == len(hypotheses), "The number of hypotheses and their reference(s) should be the " \
                                                       "same "

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Uniformly re-weighting based on maximum hypothesis lengths if largest
    # order of n-grams < 4 and weights is set at default.
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = (1 / hyp_lengths,) * hyp_lengths

    # Collects the various precision values for the different ngram orders.
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(weights, start=1)]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0

    # If there's no smoothing, set use method0 from SmoothinFunction class.
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    p_n = smoothing_function(p_n, references=references, hypothesis=hypothesis,
                             hyp_len=hyp_len)
    s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n))
    s =  bp * math.exp(math.fsum(s))
    return s


def get_sent_bleu(preds,refs):
    '''计算平均bleu，每条数据计算bleu后平均,sentence-level-bleu'''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    cc = SmoothingFunction()
    scores=[sentence_bleu(ref, pred+[' ']*max(0,2-len(pred)),weights=[0.25,0.25,0.25,0.25],smoothing_function=cc.method4)
            for pred,ref in zip(preds,refs)]
    return sum(scores)/len(scores)



class SmoothingFunction:
    """
    This is an implementation of the smoothing techniques
    for segment-level BLEU scores that was presented in
    Boxing Chen and Collin Cherry (2014) A Systematic Comparison of
    Smoothing Techniques for Sentence-Level BLEU. In WMT14.
    http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
    """

    def __init__(self, epsilon=0.1, alpha=5, k=5):
        """
        This will initialize the parameters required for the various smoothing
        techniques, the default values are set to the numbers used in the
        experiments from Chen and Cherry (2014).

        >>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures',
        ...                 'that', 'the', 'military', 'always', 'obeys', 'the',
        ...                 'commands', 'of', 'the', 'party']
        >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures',
        ...               'that', 'the', 'military', 'will', 'forever', 'heed',
        ...               'Party', 'commands']

        >>> chencherry = SmoothingFunction()
        >>> print (sentence_bleu([reference1], hypothesis1)) # doctest: +ELLIPSIS
        0.4118...
        >>> print (sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method0)) # doctest: +ELLIPSIS
        0.4118...
        >>> print (sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method1)) # doctest: +ELLIPSIS
        0.4118...
        >>> print (sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method2)) # doctest: +ELLIPSIS
        0.4489...
        >>> print (sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method3)) # doctest: +ELLIPSIS
        0.4118...
        >>> print (sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method4)) # doctest: +ELLIPSIS
        0.4118...
        >>> print (sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method5)) # doctest: +ELLIPSIS
        0.4905...
        >>> print (sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method6)) # doctest: +ELLIPSIS
        0.4135...
        >>> print (sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method7)) # doctest: +ELLIPSIS
        0.4905...

        :param epsilon: the epsilon value use in method 1
        :type epsilon: float
        :param alpha: the alpha value use in method 6
        :type alpha: int
        :param k: the k value use in method 4
        :type k: int
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k

    def method0(self, p_n, *args, **kwargs):
        """
        No smoothing.
        """
        p_n_new = []
        for i, p_i in enumerate(p_n):
            if p_i.numerator != 0:
                p_n_new.append(p_i)
            else:
                _msg = str("\nThe hypothesis contains 0 counts of {}-gram overlaps.\n"
                           "Therefore the BLEU score evaluates to 0, independently of\n"
                           "how many N-gram overlaps of lower order it contains.\n"
                           "Consider using lower n-gram order or use "
                           "SmoothingFunction()").format(i+1)
                warnings.warn(_msg)
                # When numerator==0 where denonminator==0 or !=0, the result
                # for the precision score should be equal to 0 or undefined.
                # Due to BLEU geometric mean computation in logarithm space,
                # we we need to take the return sys.float_info.min such that
                # math.log(sys.float_info.min) returns a 0 precision score.
                p_n_new.append(sys.float_info.min)
        return p_n_new

    def method1(self, p_n, *args, **kwargs):
        """
        Smoothing method 1: Add *epsilon* counts to precision with 0 counts.
        """
        return [(p_i.numerator + self.epsilon) / p_i.denominator
                if p_i.numerator == 0 else p_i for p_i in p_n]

    def method2(self, p_n, *args, **kwargs):
        """
        Smoothing method 2: Add 1 to both numerator and denominator from
        Chin-Yew Lin and Franz Josef Och (2004) Automatic evaluation of
        machine translation quality using longest common subsequence and
        skip-bigram statistics. In ACL04.
        """
        return [Fraction(p_i.numerator + 1, p_i.denominator + 1, _normalize=False) for p_i in p_n]

    def method3(self, p_n, *args, **kwargs):
        """
        Smoothing method 3: NIST geometric sequence smoothing
        The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each
        precision score whose matching n-gram count is null.
        k is 1 for the first 'n' value for which the n-gram match count is null/
        For example, if the text contains:
         - one 2-gram match
         - and (consequently) two 1-gram matches
        the n-gram count for each individual precision score would be:
         - n=1  =>  prec_count = 2     (two unigrams)
         - n=2  =>  prec_count = 1     (one bigram)
         - n=3  =>  prec_count = 1/2   (no trigram,  taking 'smoothed' value of 1 / ( 2^k ), with k=1)
         - n=4  =>  prec_count = 1/4   (no fourgram, taking 'smoothed' value of 1 / ( 2^k ), with k=2)
        """
        incvnt = 1  # From the mteval-v13a.pl, it's referred to as k.
        for i, p_i in enumerate(p_n):
            if p_i.numerator == 0:
                p_n[i] = 1 / (2 ** incvnt * p_i.denominator)
                incvnt += 1
        return p_n

    def method4(self, p_n, references, hypothesis, hyp_len, *args, **kwargs):
        """
        Smoothing method 4:
        Shorter translations may have inflated precision values due to having
        smaller denominators; therefore, we give them proportionally
        smaller smoothed counts. Instead of scaling to 1/(2^k), Chen and Cherry
        suggests dividing by 1/ln(len(T)), where T is the length of the translation.
        """
        for i, p_i in enumerate(p_n):
            if p_i.numerator == 0 and hyp_len != 0:
                incvnt = i + 1 * self.k / math.log(hyp_len)  # Note that this K is different from the K from NIST.
                p_n[i] = 1 / incvnt
        return p_n

    def method5(self, p_n, references, hypothesis, hyp_len, *args, **kwargs):
        """
        Smoothing method 5:
        The matched counts for similar values of n should be similar. To a
        calculate the n-gram matched count, it averages the n−1, n and n+1 gram
        matched counts.
        """
        m = {}
        p_n_plus1 = p_n + [modified_precision(references, hypothesis, 5)]
        m[-1] = p_n[0] + 1
        for i, p_i in enumerate(p_n):
            p_n[i] = (m[i - 1] + p_i + p_n_plus1[i + 1]) / 3
            m[i] = p_n[i]
        return p_n

    def method6(self, p_n, references, hypothesis, hyp_len, *args, **kwargs):
        """
        Smoothing method 6:
        Interpolates the maximum likelihood estimate of the precision *p_n* with
        a prior estimate *pi0*. The prior is estimated by assuming that the ratio
        between pn and pn−1 will be the same as that between pn−1 and pn−2; from
        Gao and He (2013) Training MRF-Based Phrase Translation Models using
        Gradient Ascent. In NAACL.
        """
        # This smoothing only works when p_1 and p_2 is non-zero.
        # Raise an error with an appropriate message when the input is too short
        # to use this smoothing technique.
        assert p_n[2], "This smoothing method requires non-zero precision for bigrams."
        for i, p_i in enumerate(p_n):
            if i in [0, 1]:  # Skips the first 2 orders of ngrams.
                continue
            else:
                pi0 = 0 if p_n[i - 2] == 0 else p_n[i - 1] ** 2 / p_n[i - 2]
                # No. of ngrams in translation that matches the reference.
                m = p_i.numerator
                # No. of ngrams in translation.
                l = sum(1 for _ in ngrams(hypothesis, i + 1))
                # Calculates the interpolated precision.
                p_n[i] = (m + self.alpha * pi0) / (l + self.alpha)
        return p_n

    def method7(self, p_n, references, hypothesis, hyp_len, *args, **kwargs):
        """
        Smoothing method 6:
        Interpolates the maximum likelihood estimate of the precision *p_n* with
        a prior estimate *pi0*. The prior is estimated by assuming that the ratio
        between pn and pn−1 will be the same as that between pn−1 and pn−2.
        """
        p_n = self.method4(p_n, references, hypothesis, hyp_len)
        p_n = self.method5(p_n, references, hypothesis, hyp_len)
        return p_n




def modified_precision(references, hypothesis, n):
    """
    Calculate modified ngram precision.

    The normal precision method may lead to some wrong translations with
    high-precision, e.g., the translation, in which a word of reference
    repeats several times, has very high precision.

    This function only returns the Fraction object that contains the numerator
    and denominator necessary to calculate the corpus-level precision.
    To calculate the modified precision for a single pair of hypothesis and
    references, cast the Fraction object into a float.

    The famous "the the the ... " example shows that you can get BLEU precision
    by duplicating high frequency words.

        >>> reference1 = 'the cat is on the mat'.split()
        >>> reference2 = 'there is a cat on the mat'.split()
        >>> hypothesis1 = 'the the the the the the the'.split()
        >>> references = [reference1, reference2]
        >>> float(modified_precision(references, hypothesis1, n=1)) # doctest: +ELLIPSIS
        0.2857...

    In the modified n-gram precision, a reference word will be considered
    exhausted after a matching hypothesis word is identified, e.g.

        >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
        ...               'ensures', 'that', 'the', 'military', 'will',
        ...               'forever', 'heed', 'Party', 'commands']
        >>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
        ...               'guarantees', 'the', 'military', 'forces', 'always',
        ...               'being', 'under', 'the', 'command', 'of', 'the',
        ...               'Party']
        >>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
        ...               'army', 'always', 'to', 'heed', 'the', 'directions',
        ...               'of', 'the', 'party']
        >>> hypothesis = 'of the'.split()
        >>> references = [reference1, reference2, reference3]
        >>> float(modified_precision(references, hypothesis, n=1))
        1.0
        >>> float(modified_precision(references, hypothesis, n=2))
        1.0

    An example of a normal machine translation hypothesis:

        >>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
        ...               'ensures', 'that', 'the', 'military', 'always',
        ...               'obeys', 'the', 'commands', 'of', 'the', 'party']

        >>> hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops',
        ...               'forever', 'hearing', 'the', 'activity', 'guidebook',
        ...               'that', 'party', 'direct']

        >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
        ...               'ensures', 'that', 'the', 'military', 'will',
        ...               'forever', 'heed', 'Party', 'commands']

        >>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
        ...               'guarantees', 'the', 'military', 'forces', 'always',
        ...               'being', 'under', 'the', 'command', 'of', 'the',
        ...               'Party']

        >>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
        ...               'army', 'always', 'to', 'heed', 'the', 'directions',
        ...               'of', 'the', 'party']
        >>> references = [reference1, reference2, reference3]
        >>> float(modified_precision(references, hypothesis1, n=1)) # doctest: +ELLIPSIS
        0.9444...
        >>> float(modified_precision(references, hypothesis2, n=1)) # doctest: +ELLIPSIS
        0.5714...
        >>> float(modified_precision(references, hypothesis1, n=2)) # doctest: +ELLIPSIS
        0.5882352941176471
        >>> float(modified_precision(references, hypothesis2, n=2)) # doctest: +ELLIPSIS
        0.07692...


    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hypothesis: A hypothesis translation.
    :type hypothesis: list(str)
    :param n: The ngram order.
    :type n: int
    :return: BLEU's modified precision for the nth order ngram.
    :rtype: Fraction
    """
    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    # Extract a union of references' counts.
    # max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    reference_counts[ngram])

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {ngram: min(count, max_counts[ngram])
                      for ngram, count in counts.items()}

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator, _normalize=False)


def closest_ref_length(references, hyp_len):
    """
    This function finds the reference that is the closest length to the
    hypothesis. The closest reference length is referred to as *r* variable
    from the brevity penalty formula in Papineni et. al. (2002)

    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hyp_len: The length of the hypothesis.
    :type hyp_len: int
    :return: The length of the reference that's closest to the hypothesis.
    :rtype: int
    """
    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len:
    (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len


def brevity_penalty(closest_ref_len, hyp_len):
    """
    Calculate brevity penalty.

    As the modified n-gram precision still has the problem from the short
    length sentence, brevity penalty is used to modify the overall BLEU
    score according to length.

    An example from the paper. There are three references with length 12, 15
    and 17. And a concise hypothesis of the length 12. The brevity penalty is 1.

        >>> reference1 = list('aaaaaaaaaaaa')      # i.e. ['a'] * 12
        >>> reference2 = list('aaaaaaaaaaaaaaa')   # i.e. ['a'] * 15
        >>> reference3 = list('aaaaaaaaaaaaaaaaa') # i.e. ['a'] * 17
        >>> hypothesis = list('aaaaaaaaaaaa')      # i.e. ['a'] * 12
        >>> references = [reference1, reference2, reference3]
        >>> hyp_len = len(hypothesis)
        >>> closest_ref_len =  closest_ref_length(references, hyp_len)
        >>> brevity_penalty(closest_ref_len, hyp_len)
        1.0

    In case a hypothesis translation is shorter than the references, penalty is
    applied.

        >>> references = [['a'] * 28, ['a'] * 28]
        >>> hypothesis = ['a'] * 12
        >>> hyp_len = len(hypothesis)
        >>> closest_ref_len =  closest_ref_length(references, hyp_len)
        >>> brevity_penalty(closest_ref_len, hyp_len)
        0.2635971381157267

    The length of the closest reference is used to compute the penalty. If the
    length of a hypothesis is 12, and the reference lengths are 13 and 2, the
    penalty is applied because the hypothesis length (12) is less then the
    closest reference length (13).

        >>> references = [['a'] * 13, ['a'] * 2]
        >>> hypothesis = ['a'] * 12
        >>> hyp_len = len(hypothesis)
        >>> closest_ref_len =  closest_ref_length(references, hyp_len)
        >>> brevity_penalty(closest_ref_len, hyp_len) # doctest: +ELLIPSIS
        0.9200...

    The brevity penalty doesn't depend on reference order. More importantly,
    when two reference sentences are at the same distance, the shortest
    reference sentence length is used.

        >>> references = [['a'] * 13, ['a'] * 11]
        >>> hypothesis = ['a'] * 12
        >>> hyp_len = len(hypothesis)
        >>> closest_ref_len =  closest_ref_length(references, hyp_len)
        >>> bp1 = brevity_penalty(closest_ref_len, hyp_len)
        >>> hyp_len = len(hypothesis)
        >>> closest_ref_len =  closest_ref_length(reversed(references), hyp_len)
        >>> bp2 = brevity_penalty(closest_ref_len, hyp_len)
        >>> bp1 == bp2 == 1
        True

    A test example from mteval-v13a.pl (starting from the line 705):

        >>> references = [['a'] * 11, ['a'] * 8]
        >>> hypothesis = ['a'] * 7
        >>> hyp_len = len(hypothesis)
        >>> closest_ref_len =  closest_ref_length(references, hyp_len)
        >>> brevity_penalty(closest_ref_len, hyp_len) # doctest: +ELLIPSIS
        0.8668...

        >>> references = [['a'] * 11, ['a'] * 8, ['a'] * 6, ['a'] * 7]
        >>> hypothesis = ['a'] * 7
        >>> hyp_len = len(hypothesis)
        >>> closest_ref_len =  closest_ref_length(references, hyp_len)
        >>> brevity_penalty(closest_ref_len, hyp_len)
        1.0

    :param hyp_len: The length of the hypothesis for a single sentence OR the
    sum of all the hypotheses' lengths for a corpus
    :type hyp_len: int
    :param closest_ref_len: The length of the closest reference for a single
    hypothesis OR the sum of all the closest references for every hypotheses.
    :type closest_ref_len: int
    :return: BLEU's brevity penalty.
    :rtype: float
    """
    if hyp_len > closest_ref_len:
        return 1
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)

