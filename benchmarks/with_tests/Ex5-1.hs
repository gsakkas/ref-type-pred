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
fromList :: Int -> [(Int, a)] -> Maybe (Sparse a)
fromList dim elts = fmap (SP dim) es
  where es = mapM (ok dim) elts
        ok dim (e, a) | 0 <= e && e < dim = Just (e, a)
                      | otherwise         = Nothing

{-@ test1 :: Maybe (SparseN String 3) @-}
test1 = fromList 3 [(0, "cat"), (2, "mouse")]
test2 = fromList 5 [(0, 4), (3, 1)] == Just (SP 5 [(0, 4), (3, 1)])
test3 = fromList 2 [(0, 4), (3, 1)] == Nothing