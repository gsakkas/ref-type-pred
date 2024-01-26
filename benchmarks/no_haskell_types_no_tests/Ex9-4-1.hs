module Ex9_4_1 where

{-@ LIQUID "--no-termination" @-}

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

{-@ data SList a = SL { size :: Nat, elems :: {v:[a] | realSize v = size}} @-}
data SList a = SL { size :: Int, elems :: [a] } deriving (Show, Eq)

{-@ measure realSize @-}
realSize :: [a] -> Int
realSize [] = 0
realSize (_:xs) = 1 + realSize xs

{-@ type SListN a N = {v:SList a | size v = N} @-}
{-@ type SListLE a N = {v:SList a | size v <= N} @-}

{-@ nil :: SListN a 0 @-}
nil = SL 0 []

{-@ cons :: a -> xs:SList a -> SListN a {size xs + 1} @-}
cons x (SL n xs) = SL (n+1) (x:xs)

{-@ hd :: xs:{SList a | size xs > 0} -> a @-}
hd (SL _ (x:_)) = x
hd _ = die "empty SList"

{-@ tl :: xs:{SList a | size xs > 0} -> SListN a {size xs - 1} @-}
tl (SL n (_:xs)) = SL (n-1) xs
tl _ = die "empty SList"

{-@ data Queue a = Q {front :: SList a, back :: SListLE a (size front)} @-}
data Queue a = Q { front :: SList a, back :: SList a} deriving (Show, Eq)

{-@ measure qsize @-}
qsize (Q f b) = size f + size b

{-@ type QueueN a N = {v:Queue a | qsize v = N} @-}

{-@ rot :: f:SList a -> b:{SList a | size b = 1 + size f} -> a:SList a
        -> {v:SList a | size v = size f + size b + size a}
@-}
rot f b a
  | size f == 0 = hd b `cons` a
  | otherwise   = hd f `cons` rot (tl f) (tl b) (hd b `cons` a)

{-@ makeq :: f:SList a -> b:{SList a | size b <= 1 + size f} -> QueueN a {size f + size b} @-}
makeq f b
  | size b <= size f = Q f b
  | otherwise        = Q (rot f b nil) nil

{-@ emp :: QueueN a 0 @-}
emp :: Queue a
emp = Q nil nil

{-@ insert :: <mask> @-}
insert e (Q f b) = makeq f (e `cons` b)