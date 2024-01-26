module Ex9_1_1 where

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

{-@ data SList a = SL { size :: Nat, elems :: {v:[a] | realSize v = size}} @-}
data SList a = SL { size :: Int, elems :: [a] } deriving (Show, Eq)

{-@ measure realSize @-}
realSize :: [a] -> Int
realSize [] = 0
realSize (_:xs) = 1 + realSize xs

{-@ type SListN a N = {v:SList a | size v = N} @-}

{-@ hd :: <mask> @-}
hd (SL _ (x:_)) = x
hd _ = die "empty SList"