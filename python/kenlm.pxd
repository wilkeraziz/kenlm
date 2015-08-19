cimport _kenlm


cdef bytes as_str(data)


cdef class FullScoreReturn:
    """
    Wrapper around FullScoreReturn.

    Notes:
        `prob` has been renamed to `log_prob`
        `oov` has been added to flag whether the word is OOV
    """

    cdef double log_prob
    cdef int ngram_length
    cdef bint oov


cdef class State:
    """
    Wrapper around lm::ngram::State so that python code can make incremental queries.

    Notes:
        * rich comparisons
        * hashable
    """

    cdef _kenlm.State _c_state


cdef class Model:
    """
    This is a wrapper around lm::ngram::Model.
    """

    cdef _kenlm.Model* model
    cdef public bytes path
    cdef _kenlm.const_Vocabulary* vocab

    cpdef BeginSentenceWrite(self, State state)

    cpdef NullContextWrite(self, State state)

    cpdef double BaseScore(self, State in_state, str word, State out_state)

    cpdef FullScoreReturn BaseFullScore(self, State in_state, str word, State out_state)
