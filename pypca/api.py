"""pypca.API
"""
import sklearn.decomposition as skld
from . import base as base
from . import sklearn_pca, default_sklearn_pca

# CONSTANTS FOR THIS FILE ARE FOUND IN __init__.py


def PCA(base_cls='auto', verbose=False, default=True, force=False, **kwargs):
    """API for constructing pypca.base.PCA derived class.

    Parameters
    ----------
    base_cls : str, sklearn.decomposition PCA object, list of PCA objects, optional (default: 'auto')
        Desired base classes to parse the PCA object from
    verbose : bool, optional (default: False)
        Print to console 
    default : bool, optional (default: True)
        In the case of ambiguities, resolve to defaults (TruncatedSVD and PCA)
    force : bool, optional (default: False)
        In the case of complete argument clash, choose the bases with the least clashes
    **kwargs
        implicit arguments to pass to PCA. If kwargs['pcaargs'] does not exist, 
        kwargs is parsed to only consider valid sklearn parameters. 
    kwargs['pcaargs'] : dict, optional
        Explicit declaration of arguments to consider for parsing PCA class.  If 
        kwargs['pcaargs'] exists, then the rest of kwargs are excluded from consideration.
        This allows you to pass kwargs to pca that otherwise would be filtered because they
        are not found in sklearn.pca_class.__init__.__code__.co_varnames.

    Returns
    -------
    pypca.PCA 
        derived class from a group of valid sklearn pca classes

    Raises
    ------
    NotImplementedError
        Description
    TypeError
        Description
    ValueError
        Description
    """

    # GET DEFAULTS FROM __init__.py
    skld_pcas = sklearn_pca + ['auto']
    default_pcas = default_sklearn_pca
    possible_bases = default_pcas

    # PARSE KWARGS INTO VALID LIST OF PARAMETER KEYS
    if 'pcaargs' in kwargs.keys():  # A pass through for pca arguments
        try:  # check if they are types.
            assert isinstance(kwargs['pcaargs'], dict)
        except AssertionError:
            raise TypeError("Input kwargs['pcaargs'] is not a dictionary")
        parameters = list(kwargs['pcaargs'].keys())
    else:  # Otherwise trim params to possible valids..
        valid_parameters = [ele for klass in skld_pcas[:-1]
                            for ele in klass.__init__.__code__.co_varnames]
        parameters = [parameter for parameter in kwargs.keys()
                      if parameter in valid_parameters]
        stripped = [k for k in kwargs.keys() if k not in parameters]
        if verbose and stripped != []:
            print(("Ignoring {} for implicit pca kwarg parsing. " +
                   "To include, pass dictionary kwargs['pcaargs'] " +
                   "with keys {}." +
                   "").format(stripped, stripped))

    # PARSE SPECIFIED BASE_CLS
    if base_cls != 'auto':
        if isinstance(base_cls, (list, tuple, set)):
            # Many bases
            base_cls = list(base_cls)
        else:
            # Only one base
            base_cls = [base_cls]
        # CHECK IF VALID BASE_CLS INPUT
        try:
            # Check if type. Redundant for now with line 73.
            assert all([isinstance(klass, type) for klass in base_cls])
        except AssertionError:
            if len(base_cls) == 1:
                # grammar is good
                msg = " is not a type."
            else:
                # grammar is great
                msg = " has at least one non-type element."
            raise TypeError("Input base_cls" + msg)

        try:
            # Check if they are in currently supported classes.
            # Can remove or expand skld_pcas to allow for different base classes
            klass_valid = [klass in skld_pcas for klass in base_cls]
            assert all(klass_valid)
        except AssertionError:
            # Should we just trim to the valid inputs or keep this error?
            if len(klass_valid) - sum(klass_valid) == 1:
                # grammar is good
                msg = ("Type {} is ")
            else:
                # grammar is great
                msg = ("Types {} are ")
            msg = (msg + "not supported, please pass" +
                   " one or a list of {}.")
            raise TypeError(msg.format(
                [klass for klass, tf in zip(base_cls, klass_valid) if not(tf)],
                skld_pcas))
        # IF TESTS PASSED, ASSIGN TO POSSIBLE BASES TO PARSE
        possible_bases = base_cls
    else:
        base_cls = skld_pcas[:-1]

    # LOOK FOR CLASHING KWARGS
    parameters_mismatched = []
    parameters_matched = []
    # We will report the closest mismatch to the user.
    possible_bases = []  # the list of bases with matches
    # FIND THE CLASHES
    for klass in base_cls:
        mismatch = [parameter for parameter in parameters
                    if parameter not in klass.__init__.__code__.co_varnames]
        match = [parameter for parameter in parameters if parameter not in mismatch]
        parameters_mismatched.append(mismatch)
        parameters_matched.append(match)
        if len(mismatch) == 0:
            possible_bases.append(klass)

    # PREP ERRORS FOR CLASHING KWARGS

    if possible_bases == []:
        mismatches = [len(klass) for klass in parameters_mismatched]
        matches = [-1 * (len(match)) for match in parameters_matched]
        match_mismatch_sort = [(ele[0], ele[1]) for ele in zip(mismatches, matches)]
        argsort = [(i, j) for i, j in sorted(zip(match_mismatch_sort,
                                                 range(0, len(match_mismatch_sort))), key=lambda pair: pair[0])]
        closest_klass = []
        closest_mismatch_kwargs = []
        closest_match_kwargs = []
        mismatches = []
        matches = []
        for error, ix in argsort:
            closest_klass.append(base_cls[ix])
            closest_mismatch_kwargs.append(parameters_mismatched[ix])
            closest_match_kwargs.append(parameters_matched[ix])
            mismatches.append(error[0])
            matches.append(abs(error[1]))

        i = 1
        rng = 0
        while i < len(mismatches) and mismatches[i - 1] == mismatches[i]:
            rng += 1
            i += 1

        if rng == 0:
            closest_klass_str = ("Closest match was {}.{} " +
                                 "with {} mismatches {} and " +
                                 "{} matches {}." +
            "").format(closest_klass[0].__module__,
                      closest_klass[0].__name__, mismatches[0],
                      closest_mismatch_kwargs[0], matches[0],
                      closest_match_kwargs[0])

        else:
            closest_klass_str = ("{} classes with {} " +
                                 "mismatching arguments " +
                                 "and {} matching arguments "+
            "").format(rng + 1, mismatches[0], matches[0])

            for j in range(rng):
                closest_klass_str += ("{}.{} mismatching {} and matching {}," +
                                      " ").format(closest_klass[j].__module__,
                                                  closest_klass[j].__name__,
                                                  closest_mismatch_kwargs[j],
                                                  closest_match_kwargs[j])

            closest_klass_str += ("and {}.{} mismatching {}." +
                                  "").format(closest_klass[rng].__module__,
                                             closest_klass[rng].__name__,
                                             closest_mismatch_kwargs[rng],
                                             closest_match_kwargs[rng])
            closest_klass_str = ("Closest class-mismatch pairs were "
                                 + closest_klass_str)
    # IGNORE CLASHES IF FORCE
        if force:
            # clean up the mismatched bases
            # by choosing feasible bases with the best match profile.
            possible_bases = [closest_klass[i] for i in range(0, rng + 1)]
            if verbose:
                print(("Stripping mismatching arguments from kwargs " +
                       "to select {} as possible bases." +
                       "").format(possible_bases))

            if mismatches[0] > matches[0] and mismatches[0] != 0:
                # Bad idea to choose bases when there is a disparity
                # in mismatch:match  kwargs
                error_str = ""
                for i in range(rng + 1):
                    error_str += ("{} : {}:{}, " +
                                  "").format(possible_bases[i],
                                             closest_mismatch_kwargs[i],
                                             closest_match_kwargs[i])

                raise ValueError(("Forcing base class {} " +
                                  "with {}:{} mismatching to matching " +
                                  "keyword arguments ratio not supported. " +
                                  "Respective mismatch to matching pairs" +
                                  "are" +
                                  error_str).format(possible_bases,
                                                    mismatches[0], matches[0]))
        else:
            raise ValueError("Keyword arguments clashed with " +
                             "base classes. " + closest_klass_str +
                             " To attempt to resolve ambiguities," +
                             " pass force=True. to pyca.PCA")

    # RESOLVE AMBIGUITIES BY SUBCLASSING FROM default_sklearn_pca
    if default:
        if len(possible_bases) > 1:  # fall back to defaults
            default_matches = [default_cls for default_cls in default_sklearn_pca
                               if default_cls in possible_bases]
            if default_matches == []:  # Don't have good logic here yet.
                raise NotImplementedError(
                    ("Keyword arguments produced an ambiguous base class collection. " +
                     "Possible matches are {}, " +
                     "however there is no logic to parse this combination." +
                     " Add more specific parameters " +
                     "or specify base_cls.").format(possible_bases))
            if verbose:
                print("The input parameters led to an ambiguous class" +
                      " so {} was chosen".format(default_matches))
            possible_bases = default_matches
    else:
        if verbose:
            print("Matched classes are {}".format(possible_bases))

    # PREP FINAL KWARGS AS A DEFLATED COPY OF INPUT KWARGS
    kwargs_nu = {}
    for k, v in kwargs.items():
        if k == 'pcaargs':
            {kwargs_nu[key]: value for key, value in kwargs['pcaargs'].items()}
        else:
            kwargs_nu[k] = v
    kwargs_nu['verbose'] = verbose
    kwargs_nu['base_class'] = possible_bases

    # GET A TYPE OBJECT FOR pypca.PCA WITH SUPERCLASSES kwargs_nu['base_class']
    try:
        child = base.get_pca(kwargs_nu['base_class'])
    except TypeError as e:
        # IF THERE IS A LOT OF CLASSES INCLUDED THERE WILL BE MRO ISSUES.
        # FOR NOW WE IGNORE BUT SHOULD PROBABLY HANDLE THIS BETTER
        # One common clash is sparsepca with minibatch sparsepca
        for _ in range(0, len(possible_bases)):
            try:
                # Just trim from the end.
                kwargs_nu['base_class'] = kwargs_nu['base_class'][:-1]
                child = base.get_pca(kwargs_nu['base_class'])
                break
            except Exception as e:
                pass
    if child is None:
        # We should never get here!
        raise ValueError("Unable to match a class to PCA instance")
    else:
        # Instantiate a baby pypca.base.PCA with parents kwargs_nu['base_class']!
        return child(**kwargs_nu)
