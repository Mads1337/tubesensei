from typing import TypeVar, Generic, Type, Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.base import BaseModel

ModelType = TypeVar("ModelType", bound=BaseModel)


class CRUDBase(Generic[ModelType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model

    async def get(
        self,
        db: AsyncSession,
        id: UUID,
        load_relationships: bool = False
    ) -> Optional[ModelType]:
        query = select(self.model).where(self.model.id == id)
        
        if load_relationships:
            for relationship in self.model.__mapper__.relationships:
                query = query.options(selectinload(getattr(self.model, relationship.key)))
        
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_multi(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None
    ) -> List[ModelType]:
        query = select(self.model)
        
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.where(getattr(self.model, key) == value)
        
        if order_by and hasattr(self.model, order_by):
            query = query.order_by(getattr(self.model, order_by))
        else:
            query = query.order_by(self.model.created_at.desc())
        
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        return result.scalars().all()

    async def create(
        self,
        db: AsyncSession,
        *,
        obj_in: Dict[str, Any]
    ) -> ModelType:
        db_obj = self.model(**obj_in)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def update(
        self,
        db: AsyncSession,
        *,
        id: UUID,
        obj_in: Dict[str, Any]
    ) -> Optional[ModelType]:
        query = (
            update(self.model)
            .where(self.model.id == id)
            .values(**obj_in)
            .returning(self.model)
        )
        result = await db.execute(query)
        await db.commit()
        return result.scalar_one_or_none()

    async def delete(
        self,
        db: AsyncSession,
        *,
        id: UUID
    ) -> bool:
        query = delete(self.model).where(self.model.id == id)
        result = await db.execute(query)
        await db.commit()
        return result.rowcount > 0

    async def count(
        self,
        db: AsyncSession,
        *,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        query = select(func.count()).select_from(self.model)
        
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.where(getattr(self.model, key) == value)
        
        result = await db.execute(query)
        return result.scalar()

    async def exists(
        self,
        db: AsyncSession,
        *,
        id: UUID
    ) -> bool:
        query = select(func.count()).select_from(self.model).where(self.model.id == id)
        result = await db.execute(query)
        return result.scalar() > 0

    async def get_by_field(
        self,
        db: AsyncSession,
        *,
        field_name: str,
        field_value: Any
    ) -> Optional[ModelType]:
        if not hasattr(self.model, field_name):
            raise ValueError(f"Model {self.model.__name__} has no field {field_name}")
        
        query = select(self.model).where(getattr(self.model, field_name) == field_value)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def bulk_create(
        self,
        db: AsyncSession,
        *,
        objs_in: List[Dict[str, Any]]
    ) -> List[ModelType]:
        db_objs = [self.model(**obj_in) for obj_in in objs_in]
        db.add_all(db_objs)
        await db.commit()
        
        for db_obj in db_objs:
            await db.refresh(db_obj)
        
        return db_objs

    async def bulk_update(
        self,
        db: AsyncSession,
        *,
        updates: List[Dict[str, Any]]
    ) -> int:
        if not updates:
            return 0
        
        stmt = update(self.model)
        result = await db.execute(stmt, updates)
        await db.commit()
        return result.rowcount