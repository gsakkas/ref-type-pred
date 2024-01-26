module Ex5_2 where

{-@ type Nat         = {v:Int | 0 <= v}            @-}
{-@ type Btwn Lo Hi  = {v:Int | Lo <= v && v < Hi} @-}
{-@ type SparseN a N = {v:Sparse a | spDim v == N} @-}

{-@ data Sparse a = SP { spDim   :: Nat
                       , spElems :: [(Btwn 0 spDim, a)]} @-}
data Sparse a = SP { spDim   :: Int
                   , spElems :: [(Int, a)] }
                   deriving (Show, Eq)

{-@ plus :: <mask> @-}
plus     :: (Num a) => Sparse a -> Sparse a -> Sparse a
plus x@(SP dx xs) y@(SP dy ys) = SP dx (foldr add xs ys)
  where
    add :: (Num a) => (Int, a) -> [(Int, a)] -> [(Int, a)]
    add xs [] = [xs]
    add (ix, x) ((iy, y):ys)
      | ix == iy  = (ix, x+y) : ys
      | otherwise = (iy, y) : add (ix, x) ys

{-@ test1 :: SparseN Int 3 @-}
test1 :: Sparse Int
test1 = plus vec1 vec2
    where
        vec1 = SP 3 [(0, 12), (2, 9)]
        vec2 = SP 3 [(0, 8), (1, 100)]
test1' = test1 == SP 3 [(0, 20), (2, 9), (1, 100)]