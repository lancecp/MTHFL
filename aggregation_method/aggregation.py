import aggregation_method.fedAvg as fedAvg
import aggregation_method.trim as trim
import aggregation_method.median as median
import aggregation_method.fltrust as fltrust
import aggregation_method.ROHFL as ROHFL
import aggregation_method.MTHFL as MTHFL
import aggregation_method.PartFedAvg as PartFedAvg
import aggregation_method.flame as flame
import aggregation_method.fine_agg as fine
import aggregation_method.Inverse_similarity as inverse


def norm_agg(gradients, name):

    if name == 'fedAvg':
        return fedAvg.fedAvg(gradients)
    elif name == 'trim':
        return trim.trim(gradients)
    elif name == 'median':
        return median.median(gradients)
    elif name == 'krum':
        return median.median(gradients)
    elif name == 'PartFedAvg':
        return PartFedAvg.PartFedAvg(gradients)
    elif name == 'flame':
        return flame.flame(gradients)


def fl_trust(trust_gradient, gradients):
    return fltrust.fltrust(trust_gradient, gradients)


def rohfl(reputation_score, gradients):
    return ROHFL.rohfl(reputation_score, gradients)


def mthfl(trust_gradient, gradients, reputation_score):
    return MTHFL.mthfl(trust_gradient, gradients, reputation_score)


def inverse_similarity(gradients, i):
    return inverse.inverse_similarity(gradients, i)


def fine_agg(trust_gradient, gradients):
    return fine.fine_agg(trust_gradient, gradients)



