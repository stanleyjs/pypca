import pypca as pca
import sklearn.decomposition as skld
from .load_tests import raises

def test_constructor_single_base_cls_0params_success():
    operator = pca.PCA(base_cls=skld.PCA, verbose=False)
    assert isinstance(operator, skld.PCA)
    assert not isinstance(operator, tuple(pca.sklearn_pca[1:]))

def test_constructor_multi_base_cls_0params_success():
    operator = pca.PCA(base_cls=pca.sklearn_pca[:2], verbose=False)
    for klass in pca.sklearn_pca[:2]:
        assert isinstance(operator, klass)
    assert not isinstance(operator, tuple(pca.sklearn_pca[2:]))

def test_constructor_auto_default_0_params_success():
    operator = pca.PCA(verbose=False)
    for klass in pca.sklearn_pca[:2]:
        assert isinstance(operator, klass)
    assert not(isinstance(operator, tuple(pca.sklearn_pca[2:])))

def test_constructor_auto_1_param_default_success():
    operator = pca.PCA(n_components=1,verbose=False)
    for klass in pca.sklearn_pca[:2]:
        assert isinstance(operator, klass)
    assert not isinstance(operator, tuple(pca.sklearn_pca[2:]))

def test_constructor_auto_0_param_no_default_success():
    operator = pca.PCA(verbose=False,default = False)
    tf = False
    for klass in pca.sklearn_pca:
        if (isinstance(operator,klass) 
            and klass not in pca.default_sklearn_pca):
            tf = True
    assert tf

def test_constructor_auto_1_param_no_default_success():
    operator = pca.PCA(n_components=1,verbose=False,default=False)
    tf = False
    for klass in pca.sklearn_pca:
        if (isinstance(operator,klass) 
            and klass not in pca.default_sklearn_pca):
            tf = True
    assert tf



def test_constructor_auto_2_param_multi_clash_fail():
    try:
        operator = pca.PCA(alpha=1,whiten=2,verbose=True)
    except ValueError as e:
        e = str(e)
        assert ('alpha' in e and 'whiten' in e)

def test_constructor_auto_multi_param_single_clash_fail():
    try:
        operator = pca.PCA(alpha=1,whiten=2, n_iter=1)
    except ValueError as e:
        e = str(e)
        assert ("matches ['alpha" in e and "mismatches ['whiten" in e)

def test_constructor_auto_multi_param_single_clash_force():
    test_bases = [skld.PCA, skld.TruncatedSVD,
               skld.KernelPCA,
               skld.IncrementalPCA]
    operator = pca.PCA(base_cls = test_bases, whiten=2, n_iter=1, force=True)

@raises(NotImplementedError)
def test_constructor_auto_1_param_default_fail():
    operator = pca.PCA(alpha=1)
@raises(ValueError)
def test_constructor_auto_multi_param_single_clash_force_fail_argratio():
    test_bases = [skld.SparsePCA]
    operator = pca.PCA(base_cls = test_bases, alpha=1,whiten=2, n_iter=1, force=True)

@raises(NotImplementedError)
def test_constructor_auto_1_param_default_fail():
    operator = pca.PCA(alpha=1)
        
@raises(TypeError)
def test_constructor_single_base_fail_string():
    operator = pca.PCA(base_cls='foobar')

@raises(TypeError)
def test_constructor_single_base_fail_bad_type():
    operator = pca.PCA(base_cls=list)

@raises(TypeError)
def test_constructor_list_base_fail_bad_type():
    operator = pca.PCA(base_cls=[skld.PCA,list])

@raises(TypeError)
def test_constructor_list_base_fail_bad_string():
    operator = pca.PCA(base_cls=[skld.PCA,'foo'])