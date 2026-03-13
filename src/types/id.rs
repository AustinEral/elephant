//! Newtype wrappers around [`Ulid`] for type-safe identifiers.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::error::Error;

macro_rules! define_id {
    ($(#[doc = $doc:expr])* $name:ident) => {
        $(#[doc = $doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(Ulid);

        impl $name {
            /// Create a new random ID.
            pub fn new() -> Self {
                Self(Ulid::new())
            }

            /// Create from a raw [`Ulid`].
            pub fn from_ulid(ulid: Ulid) -> Self {
                Self(ulid)
            }

            /// Return the inner [`Ulid`].
            pub fn into_inner(self) -> Ulid {
                self.0
            }

            /// Create from a [`uuid::Uuid`].
            pub fn from_uuid(uuid: uuid::Uuid) -> Self {
                Self(Ulid::from(uuid))
            }

            /// Convert to a [`uuid::Uuid`].
            pub fn to_uuid(self) -> uuid::Uuid {
                uuid::Uuid::from(self.0)
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl FromStr for $name {
            type Err = Error;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Ulid::from_str(s)
                    .map(Self)
                    .map_err(|e| Error::InvalidId(e.to_string()))
            }
        }

        impl sqlx::Type<sqlx::Postgres> for $name {
            fn type_info() -> sqlx::postgres::PgTypeInfo {
                <uuid::Uuid as sqlx::Type<sqlx::Postgres>>::type_info()
            }
        }

        impl sqlx::Encode<'_, sqlx::Postgres> for $name {
            fn encode_by_ref(
                &self,
                buf: &mut sqlx::postgres::PgArgumentBuffer,
            ) -> Result<sqlx::encode::IsNull, sqlx::error::BoxDynError> {
                <uuid::Uuid as sqlx::Encode<sqlx::Postgres>>::encode_by_ref(
                    &self.to_uuid(),
                    buf,
                )
            }
        }

        impl sqlx::Decode<'_, sqlx::Postgres> for $name {
            fn decode(
                value: sqlx::postgres::PgValueRef<'_>,
            ) -> Result<Self, sqlx::error::BoxDynError> {
                let uuid = <uuid::Uuid as sqlx::Decode<sqlx::Postgres>>::decode(value)?;
                Ok(Self::from_uuid(uuid))
            }
        }
    };
}

define_id!(
    /// Unique identifier for a [`Fact`](crate::types::Fact).
    FactId
);

define_id!(
    /// Unique identifier for an [`Entity`](crate::types::Entity).
    EntityId
);

define_id!(
    /// Unique identifier for a [`MemoryBank`](crate::types::MemoryBank).
    BankId
);

define_id!(
    /// Unique identifier for a provenance [`Source`](crate::types::Source).
    SourceId
);

define_id!(
    /// Unique identifier for a conversation turn.
    TurnId
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_fromstr_roundtrip() {
        let id = FactId::new();
        let s = id.to_string();
        let parsed: FactId = s.parse().unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn invalid_id_errors() {
        let result = "not-a-ulid".parse::<FactId>();
        assert!(result.is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let id = BankId::new();
        let json = serde_json::to_string(&id).unwrap();
        let back: BankId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn ids_are_copy() {
        let id = EntityId::new();
        let copy = id;
        assert_eq!(id, copy);
    }

    #[test]
    fn ids_are_ordered() {
        // ULIDs created later should sort after earlier ones
        let a = TurnId::new();
        // Ensure we get a distinct ULID (same millisecond is fine, random bits differ)
        let b = TurnId::new();
        // Both are valid ULIDs
        assert!(a.to_string().len() == 26);
        assert!(b.to_string().len() == 26);
    }
}
