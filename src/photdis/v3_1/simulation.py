"""Independent-source narrow radial controls with explicit native provenance."""
import numpy as np


def draw_narrow(library, contexts, distance_kpc, width_mag, *, n=128,
                seed=311712, first_identity=0):
    for table in (library,contexts):
        if 'TYPE' not in table or not (np.char.strip(table['TYPE'].astype('U'))=='PSF').all():
            raise ValueError('Verified native PSF sources and contexts required')
    if distance_kpc<=0 or width_mag<=0 or n<=0:raise ValueError('Positive radial design and sample size required')
    rng=np.random.default_rng(seed); populations=[]
    for c in range(3):
        populations.append(np.flatnonzero((library['class_label']==c)&library['mask'][:,[0,1,3]].all(axis=1)))
    weights=np.array([.75,.05,.20]);absent=np.array([not len(x) for x in populations]);weights[absent]=0
    if weights.sum()==0:raise ValueError('No admissible empirical source support')
    weights/=weights.sum();phot=[];truth=[];attempted=0;accepted=0;total=0
    center=10+5*np.log10(distance_kpc)
    while total<n:
        size=max(4*n,512);attempted+=size
        if attempted>200000:raise RuntimeError('Narrow simulation entry inefficient; retain failure')
        label=rng.choice(3,size=size,p=weights);source=np.empty(size,int)
        for c in range(3):
            use=label==c
            if use.any():source[use]=rng.choice(populations[c],size=int(use.sum()))
        context=rng.integers(len(contexts['identity']),size=size)
        mu=rng.normal(center,width_mag,size);binary=rng.random(size)<.1
        transmission=contexts['transmission'][context].copy();error=contexts['flux_error'][context].copy()
        mask=library['mask'][source]&contexts['band_mask'][context]
        flux=library['flux'][source]*10**(-.4*(mu[:,None]-10))*np.where(binary,2.,1.)[:,None]*transmission
        measured=flux+rng.normal(size=flux.shape)*np.where(mask,error,0.)
        entry=mask[:,[0,1,3]].all(axis=1)&(measured[:,[0,1,3]]>=5*error[:,[0,1,3]]).all(axis=1)
        if not entry.any():continue
        accepted+=int(entry.sum());total+=int(entry.sum())
        state=np.where(mask,np.where(measured/error>=5,4,3),0).astype('i1')
        measured[~mask]=np.nan;error[~mask]=np.nan
        values=dict(flux=measured,flux_error=error,transmission=transmission,band_mask=mask,band_state=state,
            EBV=contexts['EBV'][context],photometric_system=contexts['photometric_system'][context],
            RA=contexts['RA'][context],DEC=contexts['DEC'][context],cell=contexts['cell'][context],
            field=contexts['field'][context],TYPE=np.full(size,'PSF'),design_weight=np.ones(size))
        phot.append({k:v[entry] for k,v in values.items()})
        truth.append(dict(mu=mu[entry],class_label=label[entry],binary=binary[entry],
            template_identity=library['identity'][source][entry],context_identity=contexts['identity'][context][entry],
            corrected_feh=library['feh'][source][entry],corrected_feh_valid=library['feh_valid'][source][entry]))
    phot={k:np.concatenate([p[k] for p in phot])[:n] for k in phot[0]}
    truth={k:np.concatenate([t[k] for t in truth])[:n] for k in truth[0]}
    phot['identity']=-(910000000000000000+first_identity+np.arange(n,dtype='i8'));phot['ls_id']=phot['identity'].copy()
    return phot,truth,dict(attempted=attempted,accepted_before_cap=accepted,generated_class_weights=weights.tolist(),
        absent_classes=np.flatnonzero(absent).tolist(),true_location_mu=float(center),true_width_mu=width_mag,
        teacher_scope='One shared source-level empirical realization; repetitions are not independent real absolute-scale references')
