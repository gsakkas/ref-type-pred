module Ex9_6 where

import Prelude hiding (take)

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
emp = Q nil nil

{-@ remove :: q:{Queue a | qsize q > 0} -> (a, QueueN a {qsize q - 1}) @-}
remove :: Queue a -> (a, Queue a)
remove (Q f b) = (hd f, makeq (tl f) b)

{-@ insert :: a -> q:Queue a -> QueueN a {qsize q + 1} @-}
insert e (Q f b) = makeq f (e `cons` b)

{-@ take :: n:Nat -> q:{Queue a | qsize q >= n} -> (QueueN a n, QueueN a {qsize q - n}) @-}
take :: Int -> Queue a -> (Queue a, Queue a)
take 0 q = (emp , q)
take n q = (insert x out , q'')
    where
        (x , q') = remove q
        (out, q'') = take (n-1) q'

{-@ exampleQ1 :: QueueN _ 3 @-}
exampleQ1 = Q (SL 3 ["alice","bob","nal"]) (SL 0 [])

{-@ okTake1 :: (QueueN _ 2, QueueN _ 1) @-}
okTake1 = take 2 exampleQ1
test1 = okTake1 == (Q (SL 1 ["bob"]) (SL 1 ["alice"]), Q (SL 1 ["nal"]) (SL 0 []))

{-@ okTake2 :: (QueueN _ 3, QueueN _ 0) @-}
okTake2 = take 3 exampleQ1
test2 = okTake2 == (Q (SL 3 ["nal", "bob", "alice"]) (SL 0 []), Q (SL 0 []) (SL 0 []))