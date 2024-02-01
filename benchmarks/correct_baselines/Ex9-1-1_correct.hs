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

{-@ hd :: xs:{SList a | size xs > 0} -> a @-}
hd (SL _ (x:_)) = x
hd _ = die "empty SList"

{-@ test1 :: String @-}
test1 = hd (SL 1 ["cat"])
test1' = hd (SL 1 ["cat"]) == "cat"

{-@ test2 :: String @-}
test2 = hd (SL 2 ["cat", "dog"])
test2' = hd (SL 2 ["cat", "dog"]) == "cat"

-- {-@ test3 :: {v:_ | false} @-}
-- test3 = hd (SL 0 [])
-- test3' = hd (SL 0 []) == undefined