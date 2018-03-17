import logging
import numpy as np
log = logging.getLogger("feature_sign_search_algo")
log.setLevel(logging.INFO)


def fss(dic, sig, spar, main_ans=None):
    
    ezer = 1e-18
    
    AMAT = np.dot(dic.T, dic)

    tcorr = np.dot(dic.T, sig)

   
    if main_ans is None:
        main_ans = np.zeros(AMAT.shape[0])
    else:
        assert main_ans.ndim == 1, "main_ans must be 1-dimensional"
        assert main_ans.shape[0] == dic.shape[1], (
            "main_ans.shape[0] does not match dic.shape[1]"
        )
        
        main_ans[...] = 0.
    signs = np.zeros(AMAT.shape[0], dtype=np.int8)

    Sett = set()
    z_opt = np.inf

    nz_opt = 0

    grad = - 2 * tcorr 
    max_g = np.argmax(np.abs(grad))
    

    sds = np.dot(sig.T, sig)

    while z_opt > spar or not np.allclose(nz_opt, 0):

        if np.allclose(nz_opt, 0):
            candidate = np.argmax(np.abs(grad) * (signs == 0))
            print("cand_feature: %d" % candidate)
            if grad[candidate] > spar:
                signs[candidate] = -1.
                main_ans[candidate] = 0.
                print("add_feature %d with neg sign" %
                          candidate)
                Sett.add(candidate)
            elif grad[candidate] < -spar:
                signs[candidate] = 1.
                main_ans[candidate] = 0.
                print("add_feature %d with pos sign" %
                          candidate)
                Sett.add(candidate)
            if len(Sett) == 0:
                break
        else:
            log.debug("Non-zero coefficient optimality not satisfied, "
                      "skipping new feature activation")
        tript = np.array(sorted(Sett))
        restr_gram = AMAT[np.ix_(tript, tript)]
        restr_corr = tcorr[tript]
        restr_sign = signs[tript]
        rhs = restr_corr - spar * restr_sign / 2
        new_main_ans = np.linalg.solve(np.atleast_2d(restr_gram), rhs)
        new_signs = np.sign(new_main_ans)
        restr_oldsol = main_ans[tript]
        sign_flips = np.where(abs(new_signs - restr_sign) > 1)[0]
        print(new_signs,restr_sign,tript,new_main_ans)
        if len(sign_flips) > 0:
            best_obj = np.inf
            best_curr = None
            best_curr = new_main_ans
            best_obj = (sds + (np.dot(new_main_ans,
                                      np.dot(restr_gram, new_main_ans))
                        - 2 * np.dot(new_main_ans, restr_corr))
                        + spar * abs(new_main_ans).sum())
            if log.isEnabledFor(logging.DEBUG):


                ocost = (sds + (np.dot(restr_oldsol,
                                       np.dot(restr_gram, restr_oldsol))
                        - 2 * np.dot(restr_oldsol, restr_corr))
                        + spar * abs(restr_oldsol).sum())
                cost = (sds + np.dot(new_main_ans,
                                     np.dot(restr_gram, new_main_ans))
                        - 2 * np.dot(new_main_ans, restr_corr)
                        + spar * abs(new_main_ans).sum())
                print("Cost before linesearch (old)\t: %e" % ocost)
                print("Cost before linesearch (new)\t: %e" % cost)
            else:
                ocost = None
            for idx in sign_flips:
                a = new_main_ans[idx]
                b = restr_oldsol[idx]
                prop = b / (b - a)
                curr = restr_oldsol - prop * (restr_oldsol - new_main_ans)
                cost = sds + (np.dot(curr, np.dot(restr_gram, curr))
                              - 2 * np.dot(curr, restr_corr)
                              + spar * abs(curr).sum())
                print("Line search coefficient: %.5f cost = %e "
                          "zero-crossing coefficient's va = %e" %
                          (prop, cost, curr[idx]))
                if cost < best_obj:
                    best_obj = cost
                    best_prop = prop
                    best_curr = curr
            print("Lowest cost after linesearch\t: %e" % best_obj)
            if ocost is not None:
                if ocost < best_obj and not np.allclose(ocost, best_obj):
                    print("Warning: objective decreased from %e to %e" %
                              (ocost, best_obj))
        else:
            print("No sign flips, not doing line search")
            best_curr = new_main_ans
        main_ans[tript] = best_curr
        zeros = tript[np.abs(main_ans[tript]) < ezer]
        main_ans[zeros] = 0.
        signs[tript] = np.int8(np.sign(main_ans[tript]))
		
        Sett.difference_update(zeros)
        grad = - 2 * tcorr + 2 * np.dot(AMAT, main_ans)
        print(grad[signs == 0])
        if len(grad[signs == 0]) == 0:
             break
        z_opt = np.max(abs(grad[signs == 0]))
        nz_opt = np.max(abs(grad[signs != 0] + spar * signs[signs != 0]))
    return main_ans

sig = np.array([9,8,6,4,1])
dic = np.array([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,0]]).T
spar = 0.000001
soln = fss(dic,sig,spar)
print(soln)
print(np.matmul(dic,soln))
