module Ex5_1 where

{-@ type Nat         = {v:Int | 0 <= v}            @-}
{-@ type Btwn Lo Hi  = {v:Int | Lo <= v && v < Hi} @-}
{-@ type SparseN a N = {v:Sparse a | spDim v == N} @-}

{-@ data Sparse a = SP { spDim   :: Nat
                       , spElems :: [(Btwn 0 spDim, a)]} @-}
data Sparse a = SP { spDim   :: Int
                   , spElems :: [(Int, a)] }
                   deriving (Eq)

{-@ fromList :: <mask> @-}
fromList dim elts = fmap (SP dim) es
  where es = mapM (ok dim) elts
        ok dim (e, a) | 0 <= e && e < dim = Just (e, a)
                      | otherwise         = Nothing