from .knn import KnnMatcher
from .proxy_reranking import ProxyRerankingMatcher


def build_matcher(cfg):
    matcher = eval(cfg.MATCHER.TYPE)(cfg)
    return matcher
