module Ex11 where

import Foreign.C.Types
import Foreign.ForeignPtr
import Foreign.Ptr
import Foreign.Storable hiding (peekByteOff)
import System.IO.Unsafe
import Data.ByteString.Internal (c2w, w2c)
import Data.Word

{-@ LIQUID "--short-names"         @-}
{-@ LIQUID "--no-termination"      @-}

{-@ type TRUE = {v:Bool | v} @-}
{-@ type FALSE = {v:Bool | not v} @-}

{-@ assume mallocForeignPtrBytes :: n:Nat -> IO (ForeignPtrN a n) @-}
{-@ assume poke :: Ptr a -> a -> IO () @-}
{-@ assume peek :: OkPtr a -> IO a @-}
{-@ assume plusPtr :: p:Ptr a -> off:Nat -> v:{PtrN b {plen p - off} | 0 < plen v} @-}

{-@ peekByteOff :: OkPtr a -> Nat -> IO a @-}
peekByteOff :: (Storable a) => Ptr a -> Int -> IO a
peekByteOff p i = peek (plusPtr p i)

{-@ type Nat = {v:Int | 0 <= v} @-}
{-@ type Pos = {v:Int | 0 < v} @-}
{-@ type BNat N = {v:Nat | v <= N} @-}
{-@ type ListN a N = {v:[a] | len v = N} @-}

{-@ data ByteString = BS { bPtr :: ForeignPtr Word8,
                           bOff :: {v:Nat | v <= fplen bPtr},
                           bLen :: {v:Nat | v + bOff <= fplen bPtr}} @-}
data ByteString = BS { bPtr :: ForeignPtr Word8,
                       bOff :: !Int,
                       bLen :: !Int}

{-@ type ByteStringN N = {v:ByteString | bLen v = N} @-}
{-@ type OkPtr a = {v:Ptr a | 0 < plen v} @-}
{-@ type ByteString2 B = {v:_ | bLen (fst v) + bLen (snd v) = bLen B} @-}

{-@ create' :: n:Nat -> (PtrN Word8 n -> IO ()) -> ByteStringN n @-}
create' :: Int -> (Ptr Word8 -> IO ()) -> ByteString
create' n fill = unsafePerformIO $ do
  fp  <- mallocForeignPtrBytes n
  withForeignPtr fp fill
  return (BS fp 0 n)

{-@ pack :: s:String -> ByteStringN {len s} @-}
pack :: String -> ByteString
pack str      = create' n $ \p -> go p xs
  where
  n           = length str
  xs          = map c2w str
  go p (x:xs) = poke p x >> go (plusPtr p 1) xs
  go _ []     = return  ()

{-@ prop_unpack_length :: ByteString -> TRUE @-}
prop_unpack_length b = bLen b == length (unpack b)

{-@ unpack :: b:ByteString -> v:{String | len v = bLen b} @-}
unpack :: ByteString -> String
unpack (BS _ _ 0) = []
unpack (BS ps s l) = unsafePerformIO
                        $ withForeignPtr ps
                        $ \p -> go'' (p `plusPtr` s) (l - 1) []
    where
        {-@ go'' :: p:{_ | 0 < plen p} -> n:_ -> acc:{_ | len acc <= plen p - n} -> IO {v:_ | len v = len acc + 1 + n } @-}
        go'' p 0 acc = peekAt p 0 >>= \e -> return (w2c e : acc)
        go'' p n acc = peekAt p n >>= \e -> go'' p (n-1) (w2c e : acc)
        peekAt p n = peek (p `plusPtr` n)

{-@ unsafeTake :: n:Nat -> b:{_ | bLen b >= n} -> ByteStringN n @-}
unsafeTake :: Int -> ByteString -> ByteString
unsafeTake n (BS x s _) = BS x s n

{-@ unsafeDrop :: n:Nat -> b:{_ | bLen b >= n} -> ByteStringN {bLen b - n} @-}
unsafeDrop :: Int -> ByteString -> ByteString
unsafeDrop n (BS x s l) = BS x (s + n) (l - n)

{-@ prop_chop_length :: String -> Nat -> TRUE @-}
prop_chop_length s n
  | n <= length s = length (chop s n) == n
  | otherwise     = True

{-@ chop :: s:String -> n:BNat (len s) -> ListN Char n @-}
chop :: String -> Int -> String
chop s n = s'
    where
        b = pack s
        b' = unsafeTake n b
        s' = unpack b'

{-@ empty :: ByteStringN 0 @-}
empty = pack ""

{-@ spanByte :: Word8 -> b:ByteString -> ByteString2 b @-}
spanByte :: Word8 -> ByteString -> (ByteString, ByteString)
spanByte c ps@(BS x s ln)
    = unsafePerformIO $ withForeignPtr x $ \p -> go (p `plusPtr` s) 0
    where
        go p i
            | i >= ln   = return (ps, empty)
            | otherwise = do c' <- peekByteOff p i
                             if c /= c'
                                then return (unsafeTake i ps, unsafeDrop i ps)
                                else go p (i+1)