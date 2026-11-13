"""Conservative whole-catalog blocks for reused simulation sources and fields."""
import numpy as np
from photdis.v3_1.comparison import paired_mean


def overlap_blocks(source_sets, fields=None):
    """Join catalogs transitively when a template (or optional field) is shared.

    Disconnected components do not establish independent teacher libraries.
    These labels only describe the observed source/context reuse in this design.
    """
    sources=[set(values) for values in source_sets]
    if any(not values for values in sources):
        raise ValueError('Every catalog requires source identities')
    if fields is not None and len(fields)!=len(sources):
        raise ValueError('Aligned fields required')
    parent=list(range(len(sources)))
    def find(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]]; index=parent[index]
        return index
    def join(left,right):
        a,b=find(left),find(right); parent[max(a,b)]=min(a,b)
    seen={}; sky={}
    for i,values in enumerate(sources):
        for source in values:
            if source in seen:join(i,seen[source])
            else:seen[source]=i
        if fields is not None:
            field=fields[i]
            if field in sky:join(i,sky[field])
            else:sky[field]=i
    return np.array([find(i) for i in range(len(sources))],dtype='i8')


def block_mean(values,blocks):
    """Equal-catalog mean, resampling whole fixed blocks; singleton has no CI."""
    values=np.asarray(values,float)
    result=paired_mean(values,np.ones(len(values)),blocks,repeats=2048,seed=321201)
    return dict(n=result['n'],blocks=result['cells'],mean=result['mean'],
        lower95=result['lo95'],upper95=result['hi95'],
        interval_status='empty' if not len(values) else 'single_block_no_interval'
        if result['cells']==1 else 'conditional_whole_block_bootstrap')
